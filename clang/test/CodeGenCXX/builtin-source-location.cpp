// RUN: %clang_cc1 -std=c++2a -fblocks %s -triple x86_64-unknown-unknown -emit-llvm -o %t.ll
// RUN: %clang_cc1 -std=c++14 -fblocks %s -triple x86_64-unknown-unknown -emit-llvm -o %t.ll


// This needs to be performed before #line directives which alter filename
// RUN: %clang_cc1 -fno-file-reproducible -fmacro-prefix-map=%p=/UNLIKELY/PATH -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-PREFIX-MAP
//
// CHECK-PREFIX-MAP: /UNLIKELY/PATH{{/|\\\\}}builtin-source-location.cpp
void testRemap() {
  const char *file = __builtin_FILE();
}

#line 8 "builtin-source-location.cpp"

namespace std {
class source_location {
public:
  static constexpr source_location current(const void *__p = __builtin_source_location()) noexcept {
    source_location __loc;
    __loc.__m_impl = static_cast<const __impl *>(__p);
    return __loc;
  }
  static source_location bad_current(const void *__p = __builtin_source_location()) noexcept {
    return current(__p);
  }
  constexpr source_location() = default;
  constexpr source_location(source_location const &) = default;
  constexpr unsigned int line() const noexcept { return __m_impl->_M_line; }
  constexpr unsigned int column() const noexcept { return __m_impl->_M_column; }
  constexpr const char *file() const noexcept { return __m_impl->_M_file_name; }
  constexpr const char *function() const noexcept { return __m_impl->_M_function_name; }

private:
  // Note: The type name "std::source_location::__impl", and its constituent
  // field-names are required by __builtin_source_location().
  struct __impl {
    const char *_M_file_name;
    const char *_M_function_name;
    unsigned _M_line;
    unsigned _M_column;
  };
  const __impl *__m_impl = nullptr;
};
} // namespace std

using SL = std::source_location;

extern "C" int sink(...);

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-GLOBAL-ONE
//
// CHECK-GLOBAL-ONE-DAG: @[[FILE:.*]] = {{.*}}c"test_const_init.cpp\00"
// CHECK-GLOBAL-ONE-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
// CHECK-GLOBAL-ONE-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 1000, i32 {{[0-9]+}} }, align 8
// CHECK-GLOBAL-ONE: @const_init_global ={{.*}} global %"class.std::source_location" { ptr @[[IMPL]] }, align 8
#line 1000 "test_const_init.cpp"
SL const_init_global = SL::current();

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-GLOBAL-TWO
//
// CHECK-GLOBAL-TWO-DAG: @runtime_init_global ={{.*}} global %"class.std::source_location" zeroinitializer, align 8
//
// CHECK-GLOBAL-TWO-DAG: @[[FILE:.*]] = {{.*}}c"test_runtime_init.cpp\00"
// CHECK-GLOBAL-TWO-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
// CHECK-GLOBAL-TWO-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 1100, i32 {{[0-9]+}} }, align 8
//
// CHECK-GLOBAL-TWO: define internal void @__cxx_global_var_init()
// CHECK-GLOBAL-TWO-NOT: ret
// CHECK-GLOBAL-TWO: %call = call ptr @_ZNSt15source_location11bad_currentEPKv({{.*}} @[[IMPL]]
// CHECK-GLOBAL-TWO: store ptr %call, {{.*}} @runtime_init_global

#line 1100 "test_runtime_init.cpp"
SL runtime_init_global = SL::bad_current();

#line 2000 "test_function.cpp"
extern "C" void test_function() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-LOCAL-ONE
//
// CHECK-LOCAL-ONE-DAG: @[[FILE:.*]] = {{.*}}c"test_current.cpp\00"
// CHECK-LOCAL-ONE-DAG: @[[FUNC:.*]] = {{.*}}c"void test_function()\00"
// CHECK-LOCAL-ONE-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 2100, i32 {{[0-9]+}} }, align 8
//
// CHECK-LOCAL-ONE:  define {{.*}} @test_function
// CHECK-LOCAL-ONE:  call ptr @_ZNSt15source_location7currentEPKv({{.*}} @[[IMPL]]
#line 2100 "test_current.cpp"
  SL local = SL::current();
}

#line 3000 "TestInitClass.cpp"
struct TestInit {
  SL info = SL::current();
  SL arg_info;

#line 3100 "TestInitCtor.cpp"
  TestInit(SL arg_info = SL::current()) : arg_info(arg_info) {}
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CTOR-GLOBAL
//
// CHECK-CTOR-GLOBAL-DAG: @GlobalInitVal ={{.*}} global %struct.TestInit zeroinitializer, align 8
// CHECK-CTOR-GLOBAL-DAG: @[[FILE:.*]] = {{.*}}c"GlobalInitVal.cpp\00"
// CHECK-CTOR-GLOBAL-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
// CHECK-CTOR-GLOBAL-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 3400, i32 {{[0-9]+}} }, align 8
//
// CHECK-CTOR-GLOBAL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// CHECK-CTOR-GLOBAL-NOT: ret
//
// CHECK-CTOR-GLOBAL: call ptr @_ZNSt15source_location7currentEPKv({{.*}} @[[IMPL]]
// CHECK-CTOR-GLOBAL-NOT: ret
// CHECK-CTOR-GLOBAL: call void @_ZN8TestInitC1ESt15source_location(ptr {{[^,]*}} @GlobalInitVal, ptr
#line 3400 "GlobalInitVal.cpp"
TestInit GlobalInitVal;

extern "C" void test_init_function() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CTOR-LOCAL
//
// CHECK-CTOR-LOCAL-DAG: @[[FILE:.*]] = {{.*}}c"LocalInitVal.cpp\00"
// CHECK-CTOR-LOCAL-DAG: @[[FUNC:.*]] = {{.*}}c"void test_init_function()\00"
// CHECK-CTOR-LOCAL-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 3500, i32 {{[0-9]+}} }, align 8
//
// CHECK-CTOR-LOCAL: define{{.*}} void @test_init_function()
// CHECK-CTOR-LOCAL-NOT: ret
//
// CHECK-CTOR-LOCAL: call ptr @_ZNSt15source_location7currentEPKv({{.*}} @[[IMPL]]
// CHECK-CTOR-LOCAL-NOT: ret
// CHECK-CTOR-LOCAL: call void @_ZN8TestInitC1ESt15source_location(ptr {{[^,]*}} %init_local, ptr

#line 3500 "LocalInitVal.cpp"
  TestInit init_local;
  sink(init_local);
}

#line 4000 "ConstexprClass.cpp"
struct TestInitConstexpr {
  SL info = SL::current();
  SL arg_info;
#line 4200 "ConstexprCtor.cpp"
  constexpr TestInitConstexpr(SL arg_info = SL::current()) : arg_info(arg_info) {}
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CONSTEXPR-T2
//
// CHECK-CONSTEXPR-T2-DAG: @[[FILE_INIT:.*]] = {{.*}}c"ConstexprCtor.cpp\00"
// CHECK-CONSTEXPR-T2-DAG: @[[FUNC_INIT:.*]] = {{.*}}c"TestInitConstexpr::TestInitConstexpr(SL)\00"
// CHECK-CONSTEXPR-T2-DAG: @[[IMPL_INIT:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE_INIT]], {{[^@]*}}@[[FUNC_INIT]], i32 4200, i32 {{[0-9]+}} }, align 8
// CHECK-CONSTEXPR-T2-DAG: @[[FILE_ARG:.*]] = {{.*}}c"ConstexprGlobal.cpp\00"
// CHECK-CONSTEXPR-T2-DAG: @[[EMPTY:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
// CHECK-CONSTEXPR-T2-DAG: @[[IMPL_ARG:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE_ARG]], {{[^@]*}}@[[EMPTY]], i32 4400, i32 {{[0-9]+}} }, align 8
//
// CHECK-CONSTEXPR-T2: @ConstexprGlobal ={{.*}} global %struct.TestInitConstexpr {
// CHECK-CONSTEXPR-T2-SAME: %"class.std::source_location" { ptr @[[IMPL_INIT]] },
// CHECK-CONSTEXPR-T2-SAME: %"class.std::source_location" { ptr @[[IMPL_ARG]] }
#line 4400 "ConstexprGlobal.cpp"
TestInitConstexpr ConstexprGlobal;

extern "C" void test_init_function_constexpr() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CONSTEXPR-LOCAL
//
// CHECK-CONSTEXPR-LOCAL-DAG: @[[FUNC:.*]] = {{.*}}c"void test_init_function_constexpr()\00"
// CHECK-CONSTEXPR-LOCAL-DAG: @[[FILE:.*]] = {{.*}}c"ConstexprLocal.cpp\00"
// CHECK-CONSTEXPR-LOCAL-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 4600, i32 {{[0-9]+}} }, align 8
//
// CHECK-CONSTEXPR-LOCAL: define{{.*}} void @test_init_function_constexpr()
// CHECK-CONSTEXPR-LOCAL-NOT: ret
// CHECK-CONSTEXPR-LOCAL: call ptr @_ZNSt15source_location7currentEPKv({{.*}} @[[IMPL]]
// CHECK-CONSTEXPR-LOCAL-NOT: ret
// CHECK-CONSTEXPR-LOCAL: call void @_ZN17TestInitConstexprC1ESt15source_location(ptr {{[^,]*}} %local_val, ptr
#line 4600 "ConstexprLocal.cpp"
  TestInitConstexpr local_val;
}

#line 5000 "TestInitAgg.cpp"
struct TestInitAgg {
#line 5100 "i1.cpp"
  SL i1;
#line 5200 "i2.cpp"
  SL i2 = SL::current();
#line 5300 "TestInitAggEnd.cpp"
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-DEFAULT
//
// CHECK-AGG-DEFAULT-DAG: @[[FILE:.*]] = {{.*}}c"TestInitAgg.cpp\00"
// CHECK-AGG-DEFAULT-DAG: @[[FUNC:.*]] = {{.*}}c"TestInitAgg::TestInitAgg()\00"
// CHECK-AGG-DEFAULT-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 5000, i32 {{[0-9]+}} }, align 8
//
// CHECK-AGG-DEFAULT: @GlobalAggDefault ={{.*}} global %struct.TestInitAgg {
// CHECK-AGG-DEFAULT-SAME: %"class.std::source_location" zeroinitializer,
// CHECK-AGG-DEFAULT-SAME: %"class.std::source_location" { ptr @[[IMPL]] }
#line 5400 "GlobalAggDefault.cpp"
TestInitAgg GlobalAggDefault;

#line 5500 "test_agg_init_test.cpp"
extern "C" void test_agg_init() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-INIT

// CHECK-AGG-INIT-DAG: @[[FUNC:.*]] = {{.*}}c"void test_agg_init()\00"

// CHECK-AGG-INIT-DAG: @[[FILE:.*]] = {{.*}}c"BraceInitEnd.cpp\00"
// CHECK-AGG-INIT-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 5700, i32 {{[0-9]+}} }, align 8

#line 5600 "BraceInitStart.cpp"
  TestInitAgg local_brace_init{
#line 5700 "BraceInitEnd.cpp"
  };

// CHECK-AGG-INIT-DAG: @[[FILE:.*]] = {{.*}}c"EqualInitEnd.cpp\00"
// CHECK-AGG-INIT-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 5900, i32 {{[0-9]+}} }, align 8

#line 5800 "EqualInitStart.cpp"
  TestInitAgg local_equal_init =
      {
#line 5900 "EqualInitEnd.cpp"
      };


// CHECK-AGG-INIT-DAG: @[[FILE_DEFAULT:.*]] = {{.*}}c"InitListEnd.cpp\00"
// CHECK-AGG-INIT-DAG: @[[FILE_ELEM:.*]] = {{.*}}c"ListElem.cpp\00"
// CHECK-AGG-INIT-DAG: @[[IMPL_DEFAULT:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE_ELEM]], {{[^@]*}}@[[FUNC]], i32 6100, i32 {{[0-9]+}} }, align 8
// CHECK-AGG-INIT-DAG: @[[IMPL_ELEM:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE_DEFAULT]], {{[^@]*}}@[[FUNC]], i32 6200, i32 {{[0-9]+}} }, align 8

#line 6000 "InitListStart.cpp"
  TestInitAgg local_list_init =
      {
#line 6100 "ListElem.cpp"
          {SL::current()}
#line 6200 "InitListEnd.cpp"
      };
}

#line 7000 "TestTemplate.cpp"
template <class Tp, int>
struct TestTemplate {
  Tp info = Tp::current();
  Tp arg_info;
#line 7100 "TestTemplateCtor.cpp"
  constexpr TestTemplate(Tp arg_info = Tp::current()) : arg_info(arg_info) {}
};

#line 7200 "test_template.cpp"
template <class T, int V>
void test_template() {

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-TEMPL -DINT_ID=0
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-TEMPL -DINT_ID=1
//
// CHECK-TEMPL-DAG: @[[FILE:.*]] = {{.*}}c"local_templ.cpp\00"
// CHECK-TEMPL-DAG: @[[FUNC:.*]] = {{.*}}c"void test_template() [T = std::source_location, V = [[INT_ID]]]\00"
// CHECK-TEMPL-DAG: @[[IMPL:.*]] = private unnamed_addr constant %"struct.std::source_location::__impl" { {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]], i32 7300, i32 {{[0-9]+}} }, align 8
#line 7300 "local_templ.cpp"
  TestTemplate<T, V> local_templ;
}
#line 7400 "EndTestTemplate.cpp"
template void test_template<SL, 0>();
template void test_template<SL, 1>();
