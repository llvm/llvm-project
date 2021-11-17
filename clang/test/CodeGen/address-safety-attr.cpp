int DefinedInDifferentFile(int *a);
// RUN: echo "int DefinedInDifferentFile(int *a) { return *a; }" > %t.extra-source.cpp
// RUN: echo "struct S { S(){} ~S(){} };" >> %t.extra-source.cpp
// RUN: echo "S glob_array[5];" >> %t.extra-source.cpp

// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin -disable-O0-optnone -emit-llvm -o - %s -include %t.extra-source.cpp | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin -disable-O0-optnone -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address | FileCheck -check-prefix=ASAN %s

// RUN: echo "fun:*IgnorelistedFunction*" > %t.func.ignorelist
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin -disable-O0-optnone -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.func.ignorelist | FileCheck -check-prefix=BLFUNC %s

// The ignorelist file uses regexps, so escape backslashes, which are common in
// Windows paths.
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t.file.ignorelist
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin -disable-O0-optnone -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.file.ignorelist | FileCheck -check-prefix=BLFILE %s

// The sanitize_address attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_sanitize_address attribute
// is present.

// Attributes for function defined in different source file:
// WITHOUT: DefinedInDifferentFile{{.*}} [[NOATTR:#[0-9]+]]
// BLFILE:  DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]
// BLFUNC:  DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]
// ASAN:    DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]

// Check that functions generated for global in different source file are
// not ignorelisted.
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// WITHOUT: @__cxx_global_array_dtor{{.*}}[[NOATTR]]
// BLFILE: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]
// BLFILE: @__cxx_global_array_dtor{{.*}}[[WITH]]
// BLFUNC: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]
// BLFUNC: @__cxx_global_array_dtor{{.*}}[[WITH]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]
// ASAN: @__cxx_global_array_dtor{{.*}}[[WITH]]

// WITHOUT:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// BLFILE:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// BLFUNC:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// ASAN:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_address))
int NoAddressSafety1(int *a) { return *a; }

// WITHOUT:  NoAddressSafety2{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety2{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety2{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_address))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// WITHOUT:  NoAddressSafety3{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety3{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety3{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety3{{.*}}) [[NOATTR]]
[[gnu::no_sanitize_address]]
int NoAddressSafety3(int *a) { return *a; }

// WITHOUT:  NoAddressSafety4{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety4{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety4{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety4{{.*}}) [[NOATTR]]
[[gnu::no_sanitize_address]]
int NoAddressSafety4(int *a);
int NoAddressSafety4(int *a) { return *a; }

// WITHOUT:  NoAddressSafety5{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety5{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety5{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety5{{.*}}) [[NOATTR]]
__attribute__((no_sanitize("address")))
int NoAddressSafety5(int *a) { return *a; }

// WITHOUT:  NoAddressSafety6{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety6{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety6{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety6{{.*}}) [[NOATTR]]
__attribute__((no_sanitize("address")))
int NoAddressSafety6(int *a);
int NoAddressSafety6(int *a) { return *a; }

// WITHOUT:  AddressSafetyOk{{.*}}) [[NOATTR]]
// BLFILE:  AddressSafetyOk{{.*}}) [[NOATTR]]
// BLFUNC: AddressSafetyOk{{.*}}) [[WITH:#[0-9]+]]
// ASAN: AddressSafetyOk{{.*}}) [[WITH:#[0-9]+]]
int AddressSafetyOk(int *a) { return *a; }

// WITHOUT:  IgnorelistedFunction{{.*}}) [[NOATTR]]
// BLFILE:  IgnorelistedFunction{{.*}}) [[NOATTR]]
// BLFUNC:  IgnorelistedFunction{{.*}}) [[NOATTR]]
// ASAN:  IgnorelistedFunction{{.*}}) [[WITH]]
int IgnorelistedFunction(int *a) { return *a; }

#define GENERATE_FUNC(name) \
    int name(int *a) { return *a; }
// WITHOUT: GeneratedFunction{{.*}}) [[NOATTR]]
// BLFILE:  GeneratedFunction{{.*}}) [[NOATTR]]
// BLFUNC:  GeneratedFunction{{.*}}) [[WITH]]
// ASAN:    GeneratedFunction{{.*}}) [[WITH]]
GENERATE_FUNC(GeneratedFunction)

#define GENERATE_NAME(name) name##_generated
// WITHOUT: Function_generated{{.*}}) [[NOATTR]]
// BLFILE:  Function_generated{{.*}}) [[NOATTR]]
// BLFUNC:  Function_generated{{.*}}) [[WITH]]
// ASAN:    Function_generated{{.*}}) [[WITH]]
int GENERATE_NAME(Function)(int *a) { return *a; }

// WITHOUT:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// BLFILE:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// BLFUNC:  TemplateAddressSafetyOk{{.*}}) [[WITH]]
// ASAN: TemplateAddressSafetyOk{{.*}}) [[WITH]]
template<int i>
int TemplateAddressSafetyOk() { return i; }

// WITHOUT:  TemplateNoAddressSafety1{{.*}}) [[NOATTR]]
// BLFILE:  TemplateNoAddressSafety1{{.*}}) [[NOATTR]]
// BLFUNC:  TemplateNoAddressSafety1{{.*}}) [[NOATTR]]
// ASAN: TemplateNoAddressSafety1{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize_address))
int TemplateNoAddressSafety1() { return i; }

// WITHOUT:  TemplateNoAddressSafety2{{.*}}) [[NOATTR]]
// BLFILE:  TemplateNoAddressSafety2{{.*}}) [[NOATTR]]
// BLFUNC:  TemplateNoAddressSafety2{{.*}}) [[NOATTR]]
// ASAN: TemplateNoAddressSafety2{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize("address")))
int TemplateNoAddressSafety2() { return i; }

int force_instance = TemplateAddressSafetyOk<42>()
                   + TemplateNoAddressSafety1<42>()
                   + TemplateNoAddressSafety2<42>();

// Check that __cxx_global_var_init* get the sanitize_address attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// BLFILE: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// BLFUNC: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]

// WITHOUT: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// BLFILE: attributes [[WITH]] = { noinline nounwind sanitize_address{{.*}} }
// BLFILE: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// BLFUNC: attributes [[WITH]] = { noinline nounwind sanitize_address{{.*}} }
// BLFUNC: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }

// ASAN: attributes [[WITH]] = { noinline nounwind sanitize_address{{.*}} }
// ASAN: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }
