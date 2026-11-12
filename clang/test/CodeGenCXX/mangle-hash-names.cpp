// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fhash-mangled-names -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=HASHDIFF
// RUN: llvm-cxxfilt < %t.ll | FileCheck %s --check-prefix=DEMANGLE
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=NOHASH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -x c++-header -emit-pch %s -o %t.nohash.pch
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fhash-mangled-names -include-pch %t.nohash.pch -emit-llvm %s -o - | FileCheck %s --check-prefix=PCH-HASH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -x c++-header -fhash-mangled-names -emit-pch %s -o %t.hash.pch
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -include-pch %t.hash.pch -emit-llvm %s -o - | FileCheck %s --check-prefix=PCH-NOHASH
// RUN: %clang_cc1 -x cuda -fcuda-is-device -std=c++20 -triple nvptx64-nvidia-cuda -fhash-mangled-names -emit-llvm %s -o - | FileCheck %s --check-prefix=CUDA --implicit-check-not=.h.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-apple-macosx11.0.0 -fhash-mangled-names -emit-llvm %s -o - | FileCheck %s --check-prefix=MACHO --implicit-check-not=.h.
// RUN: printf 'fun:_ZN1a1b1c6MyFuncI1BEE1TI1AEP1PI2C1EPS7_I2C2EPS7_I2C3EPS7_I2C4EPS7_I2C5EPS7_I2C6EPS7_I2C7EPS7_I2C8E\nfun:*consumeStdStrings*\nfun:_ZN1a1b1c17consumeStdStringsERKSbIDsSt11char_traitsIDsESaIDsEEP1PI2C1EPS8_I2C2EPS8_I2C3EPS8_I2C4E=sanitize\nglobal:_ZN1a1b1c17GlobalVarTemplateB14custom_abi_tagI1PI2C1ES3_I2C2ES3_I2C3ES3_I2C4EEE\n' > %t.san.ignorelist
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fhash-mangled-names -fsanitize=memory,address -fsanitize-ignorelist=%t.san.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=SANLIST
// RUN: printf '[always]\nfun:_ZN1a1b1c6MyFuncI1BEE1TI1AEP1PI2C1EPS7_I2C2EPS7_I2C3EPS7_I2C4EPS7_I2C5EPS7_I2C6EPS7_I2C7EPS7_I2C8E\n[never]\nfun:*MyFunc*\n' > %t.xray.list
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fhash-mangled-names -fxray-instrument -fxray-attr-list=%t.xray.list -emit-llvm %s -o - | FileCheck %s --check-prefix=XRAYLIST
// RUN: printf '[llvm]\nfunction:_ZN1a1b1c6MyFuncI1BEE1TI1AEP1PI2C1EPS7_I2C2EPS7_I2C3EPS7_I2C4EPS7_I2C5EPS7_I2C6EPS7_I2C7EPS7_I2C8E=allow\nfunction:*MyFunc*=forbid\n' > %t.prof.list
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fhash-mangled-names -fprofile-instrument=llvm -fprofile-list=%t.prof.list -emit-llvm %s -o - | FileCheck %s --check-prefix=PROFLIST

#ifndef HEADER
#define HEADER

template <typename> struct T {};
struct A {};
struct B {};
struct B2 {};
template <typename> struct P {};
struct C1 {};
struct C2 {};
struct C3 {};
struct C4 {};
struct C5 {};
struct C6 {};
struct C7 {};
struct C8 {};

namespace a {
namespace b {
namespace c {

// 1. Function template with long return type and parameter list:
//    Drops template args, return type, and parameters into '.h.<uint64>' BLAKE3 hash.
template <typename U>
T<A> MyFunc(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *, P<C6> *, P<C7> *,
            P<C8> *) {
  return {};
}

template T<A> MyFunc<B>(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *, P<C6> *,
                        P<C7> *, P<C8> *);
template T<A> MyFunc<B2>(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *, P<C6> *,
                         P<C7> *, P<C8> *);

// 2. Short mangled name stays unhashed because shortened + suffix >= original.
int shortFn(int x) { return x; }

// 3. Nested class templates, method qualifiers, ctors, dtors, conversion ops,
//    virtual tables, typeinfo, and thunks.
template <typename X, typename Y> struct Outer {
  template <typename U, typename V> struct Inner {
    int method(P<C1> *, P<C2> *, P<C3> *, P<C4> *) const & { return 0; }
    int method(P<C1> *, P<C2> *, P<C3> *, P<C4> *) && { return 1; }

    template <typename W>
    Inner(W, P<C1> *, P<C2> *, P<C3> *, P<C4> *) {}

    operator P<C1>() const { return {}; }

    template <typename W>
    operator P<W>() const { return {}; }

    operator Outer<P<C5>, P<C6>>() const { return {}; }
  };
};

void instantiateNested(P<C1> *p1, P<C2> *p2, P<C3> *p3, P<C4> *p4) {
  Outer<C1, C2>::Inner<C3, C4> obj(B{}, p1, p2, p3, p4);
  const auto &cref = obj;
  (void)cref.method(p1, p2, p3, p4);
  (void)static_cast<Outer<C1, C2>::Inner<C3, C4> &&>(obj).method(p1, p2, p3, p4);
  (void)static_cast<P<C1>>(cref);
  (void)cref.operator P<C2>();
  (void)static_cast<Outer<P<C5>, P<C6>>>(cref);
}

// 4. Local static variable and guard variable inside a long template function.
int makeVal();
template <typename U>
int WithLocalStatic(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *, P<C6> *) {
  static int cachedValue = makeVal();
  auto genericLambda = []<typename L>(L, P<C1> *, P<C2> *, P<C3> *, P<C4> *) {
    return 42;
  };
  return cachedValue + genericLambda(U{}, nullptr, nullptr, nullptr, nullptr);
}

template int WithLocalStatic<B>(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *,
                                P<C6> *);

// 5. Multiple inheritance virtual thunk, vtable, and RTTI.
struct Base1 {
  virtual ~Base1() = default;
  virtual void first() {}
};
struct Base2 {
  virtual ~Base2() = default;
  virtual void vfunc(P<C1> *, P<C2> *, P<C3> *, P<C4> *) = 0;
};
template <typename U, typename V, typename W>
struct MultiDerived : Base1, Base2 {
  void vfunc(P<C1> *, P<C2> *, P<C3> *, P<C4> *) override {}
};

void useMultiDerived() {
  MultiDerived<P<C1>, P<C2>, P<C3>> d;
  d.vfunc(nullptr, nullptr, nullptr, nullptr);
}

// 6. Variable template and ABI tags.
template <typename U, typename V, typename W, typename Z>
int __attribute__((abi_tag("custom_abi_tag"))) GlobalVarTemplate = 7;

int useVarTemplate() {
  return GlobalVarTemplate<P<C1>, P<C2>, P<C3>, P<C4>>;
}

// 7. Class-member operator new / operator delete SHOULD still be hashed.
struct WithMemberAlloc {
  static void *operator new(decltype(sizeof(0)), P<C1> *, P<C2> *, P<C3> *,
                            P<C4> *);
  static void operator delete(void *, P<C1> *, P<C2> *, P<C3> *, P<C4> *);
};

} // namespace c
} // namespace b
} // namespace a

namespace std {
enum class align_val_t : decltype(sizeof(0)) {};
struct nothrow_t {
  explicit nothrow_t() = default;
};
extern const nothrow_t nothrow;

template <typename CharT> struct char_traits {};
template <typename T> struct allocator {};

template <typename CharT, typename Traits = char_traits<CharT>>
struct basic_ios {
  virtual ~basic_ios() = default;
  virtual void clear(P<C1> *, P<C2> *, P<C3> *, P<C4> *) {}
};

template <typename CharT, typename Traits, typename Alloc>
struct basic_string {
  void append(P<C1> *, P<C2> *, P<C3> *, P<C4> *, P<C5> *, P<C6> *) {}
};

inline namespace __1 {
template <typename CharT> struct inline_basic_streambuf {
  virtual ~inline_basic_streambuf() = default;
  virtual void syncbuf(P<C1> *, P<C2> *, P<C3> *, P<C4> *) {}
};
} // namespace __1
} // namespace std

namespace __gnu_debug {
struct _Safe_sequence_base {};
struct _Safe_iterator_base {
  void _M_attach(_Safe_sequence_base *, bool);
};
} // namespace __gnu_debug

namespace a {
namespace b {
namespace c {
void consumeStdStrings(
    const std::basic_string<char16_t, std::char_traits<char16_t>,
                            std::allocator<char16_t>> &,
    P<C1> *, P<C2> *, P<C3> *, P<C4> *) {}
} // namespace c
} // namespace b
} // namespace a

// 8. Global ::operator new / ::operator new[] / ::operator delete /
//    ::operator delete[] (including aligned/nothrow and custom global
//    placement/tagged overloads) MUST stay unhashed.
enum class custom_alloc_tag_t : unsigned char {};

void *operator new(decltype(sizeof(0)), std::align_val_t,
                   const std::nothrow_t &) noexcept;
void *operator new[](decltype(sizeof(0)), std::align_val_t,
                     const std::nothrow_t &) noexcept;
void operator delete(void *, std::align_val_t,
                     const std::nothrow_t &) noexcept;
void operator delete[](void *, std::align_val_t,
                       const std::nothrow_t &) noexcept;
extern "C++" {
void *operator new(decltype(sizeof(0)), std::align_val_t,
                   const std::nothrow_t &, custom_alloc_tag_t) noexcept;
void *operator new[](decltype(sizeof(0)), std::align_val_t,
                     const std::nothrow_t &, custom_alloc_tag_t) noexcept;
}

void sinkPtr(void *);

void useAllocators() {
  void *p1 = ::operator new(8, std::align_val_t{16}, std::nothrow);
  sinkPtr(p1);
  ::operator delete(p1, std::align_val_t{16}, std::nothrow);

  void *p2 = ::operator new[](16, std::align_val_t{32}, std::nothrow);
  sinkPtr(p2);
  ::operator delete[](p2, std::align_val_t{32}, std::nothrow);

  sinkPtr(::operator new(8, std::align_val_t{16}, std::nothrow,
                         custom_alloc_tag_t{1}));
  sinkPtr(::operator new[](16, std::align_val_t{32}, std::nothrow,
                           custom_alloc_tag_t{2}));

  void *pm = a::b::c::WithMemberAlloc::operator new(
      sizeof(a::b::c::WithMemberAlloc), nullptr, nullptr, nullptr, nullptr);
  sinkPtr(pm);
  a::b::c::WithMemberAlloc::operator delete(pm, nullptr, nullptr, nullptr,
                                            nullptr);
}

void useStdString() {
  using LongElem = a::b::c::MultiDerived<P<C1>, P<C2>, P<C3>>;
  std::basic_ios<LongElem> ios;
  ios.clear(nullptr, nullptr, nullptr, nullptr);
  std::inline_basic_streambuf<LongElem> sbuf;
  sbuf.syncbuf(nullptr, nullptr, nullptr, nullptr);
  std::basic_string<char16_t, std::char_traits<char16_t>,
                    std::allocator<char16_t>>
      s;
  s.append(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  __gnu_debug::_Safe_iterator_base safe_it;
  safe_it._M_attach(nullptr, false);
  a::b::c::consumeStdStrings(s, nullptr, nullptr, nullptr, nullptr);
}

// CUDA __global__ kernels and __device__ functions on NVPTX MUST stay unhashed
// so host launch stubs match external device compilers (e.g. nvcc) and PTX
// symbol names never contain '.' (rejected by ptxas).
#ifdef __CUDA__
template <typename T>
__attribute__((device)) int myCudaDeviceHelper(P<C1> *, P<C2> *, P<C3> *,
                                               P<C4> *, T *) {
  return 0;
}
template <typename T>
__attribute__((global)) void myCudaTemplateKernel(P<C1> *p1, P<C2> *p2,
                                                  P<C3> *p3, P<C4> *p4, T *t) {
  (void)myCudaDeviceHelper<T>(p1, p2, p3, p4, t);
}
template __attribute__((global)) void
myCudaTemplateKernel<C5>(P<C1> *, P<C2> *, P<C3> *, P<C4> *, C5 *);
#endif

#else

T<A> useFromPCH() {
  return a::b::c::MyFunc<int>(nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr);
}

#endif

// --- Verify IR symbols with -fhash-mangled-names ---

// Guard and local static variable:
// CHECK-DAG: @_ZZN1a1b1c15WithLocalStaticEvE11cachedValue{{[.]h[.][0-9]+[ ]}}= linkonce_odr global
// CHECK-DAG: @_ZGVZN1a1b1c15WithLocalStaticEvE11cachedValue{{[.]h[.][0-9]+[ ]}}= linkonce_odr global

// Variable template with ABI tag stripped into hash:
// CHECK-DAG: @_ZN1a1b1c17GlobalVarTemplateE{{[.]h[.][0-9]+[ ]}}= linkonce_odr global

// VTable and RTTI structure are hashed, while _ZTS (typeinfo name string) is NOT hashed:
// CHECK-DAG: @_ZTVN1a1b1c12MultiDerivedE{{[.]h[.][0-9]+[ ]}}= linkonce_odr constant
// CHECK-DAG: @_ZTSN1a1b1c12MultiDerivedI1PI2C1ES3_I2C2ES3_I2C3EEE = linkonce_odr constant
// CHECK-DAG: @_ZTIN1a1b1c12MultiDerivedE{{[.]h[.][0-9]+[ ]}}= linkonce_odr constant

// Distinct template instantiations get distinct hashes:
// CHECK-DAG: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+[(]}}
// HASHDIFF: define weak_odr void @_ZN1a1b1c6MyFuncEv.h.[[HASH_B:[0-9]+]](
// HASHDIFF-NOT: @_ZN1a1b1c6MyFuncEv.h.[[HASH_B]](

// Short name is untouched:
// CHECK-DAG: define{{.*}} i32 @_ZN1a1b1c7shortFnEi(i32

// Nested class templates, constructor template, qualified methods, conversion ops:
// CHECK-DAG: define linkonce_odr void @_ZN1a1b1c5Outer5InnerC2Ev{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: define linkonce_odr{{.*}} i32 @_ZN1a1b1c5Outer5Inner6methodEv{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: define linkonce_odr void @_ZN1a1b1c5Outer5Innercv1PEv{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: define linkonce_odr void @_ZN1a1b1c5Outer5InnercvNS1_5OuterEEv{{[.]h[.][0-9]+[(]}}

// Generic lambda inside long template function:
// CHECK-DAG: define linkonce_odr{{.*}} i32 @_ZZN1a1b1c15WithLocalStaticEvENUlvE_clEv{{[.]h[.][0-9]+[(]}}

// Virtual method and non-virtual thunk:
// CHECK-DAG: define linkonce_odr void @_ZN1a1b1c12MultiDerived5vfuncEv{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: define linkonce_odr void @_ZThn8_N1a1b1c12MultiDerived5vfuncEv{{[.]h[.][0-9]+[(]}}

// Declarations inside top-level namespace std (including inline std::__1,
// methods, and vtables) and __gnu_debug remain unhashed, while functions
// outside std taking std::basic_string args ARE hashed:
// CHECK-DAG: @_ZTVSt9basic_iosIN1a1b1c12MultiDerivedI1PI2C1E{{[A-Za-z0-9_]+}}EE = linkonce_odr constant
// CHECK-DAG: @_ZTVNSt3__122inline_basic_streambufIN1a1b1c12MultiDerivedI1PI2C1E{{[A-Za-z0-9_]+}}EE = linkonce_odr constant
// CHECK-DAG: define linkonce_odr void @_ZNSt3__122inline_basic_streambufIN1a1b1c12MultiDerivedI1PI2C1E{{[A-Za-z0-9_]+}}E7syncbufEP{{[A-Za-z0-9_]+}}E(
// CHECK-DAG: define linkonce_odr void @_ZNSbIDsSt11char_traitsIDsESaIDsEE6appendEP1PI2C1E{{[A-Za-z0-9_]+}}E(
// CHECK-DAG: declare{{.*}} void @_ZN11__gnu_debug19_Safe_iterator_base9_M_attachEPNS_19_Safe_sequence_baseEb(
// CHECK-DAG: define{{.*}} void @_ZN1a1b1c17consumeStdStringsEv{{[.]h[.][0-9]+[(]}}

// Global allocation/deallocation functions remain unhashed even when long:
// CHECK-DAG: declare{{.*}} ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(
// CHECK-DAG: declare{{.*}} ptr @_ZnamSt11align_val_tRKSt9nothrow_t(
// CHECK-DAG: declare{{.*}} void @_ZdlPvSt11align_val_tRKSt9nothrow_t(
// CHECK-DAG: declare{{.*}} void @_ZdaPvSt11align_val_tRKSt9nothrow_t(
// CHECK-DAG: declare{{.*}} ptr @_ZnwmSt11align_val_tRKSt9nothrow_t18custom_alloc_tag_t(
// CHECK-DAG: declare{{.*}} ptr @_ZnamSt11align_val_tRKSt9nothrow_t18custom_alloc_tag_t(

// Class-member allocation/deallocation functions ARE hashed:
// CHECK-DAG: declare{{.*}} ptr @_ZN1a1b1c15WithMemberAllocnwEv{{[.]h[.][0-9]+[(]}}
// CHECK-DAG: declare{{.*}} void @_ZN1a1b1c15WithMemberAllocdlEv{{[.]h[.][0-9]+[(]}}

// SamplePGO suffix elision policy is set to "selected" so '.h.<uint64>' is preserved:
// CHECK-DAG: attributes #{{[0-9]+}} = { {{.*}}"sample-profile-suffix-elision-policy"="selected"{{.*}} }

// CUDA-DAG: define{{.*}} ptx_kernel void @_Z20myCudaTemplateKernelI2C5EvP1PI2C1EPS1_I2C2EPS1_I2C3EPS1_I2C4EPT_(
// CUDA-DAG: define linkonce_odr{{.*}} i32 @_Z18myCudaDeviceHelperI2C5EiP1PI2C1EPS1_I2C2EPS1_I2C3EPS1_I2C4EPT_(

// Non-ELF targets (e.g. Apple Mach-O) keep standard dot-free Itanium mangling:
// MACHO: define weak_odr void @_ZN1a1b1c6MyFuncI1BEE1TI1AEP1PI2C1EPS7_I2C2EPS7_I2C3EPS7_I2C4EPS7_I2C5EPS7_I2C6EPS7_I2C7EPS7_I2C8E(

// --- Verify end-to-end demangling via unmodified llvm-cxxfilt ---
// DEMANGLE-DAG: a::b::c::WithLocalStatic()::cachedValue ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: guard variable for a::b::c::WithLocalStatic()::cachedValue ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::GlobalVarTemplate ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: vtable for a::b::c::MultiDerived ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: typeinfo for a::b::c::MultiDerived ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::MyFunc() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::Outer::Inner::Inner() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::Outer::Inner::method() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::Outer::Inner::operator P() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::Outer::Inner::operator a::b::c::Outer() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: non-virtual thunk to a::b::c::MultiDerived::vfunc() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::consumeStdStrings() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::WithMemberAlloc::operator new() ({{[.]h[.][0-9]+[)]}}
// DEMANGLE-DAG: a::b::c::WithMemberAlloc::operator delete() ({{[.]h[.][0-9]+[)]}}

// --- Verify default mode (flag off) keeps standard Itanium mangling ---
// NOHASH-DAG: define weak_odr void @_ZN1a1b1c6MyFuncI1BEE1TI1AEP1PI2C1EPS7_I2C2EPS7_I2C3EPS7_I2C4EPS7_I2C5EPS7_I2C6EPS7_I2C7EPS7_I2C8E(
// NOHASH-NOT: {{[.]h[.][0-9]+[ (]}}
// NOHASH-NOT: "sample-profile-suffix-elision-policy"

// --- Verify Benign PCH/module compatibility across mismatched -fhash-mangled-names ---
// PCH-HASH: define linkonce_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+[(]}}
// PCH-NOHASH: define linkonce_odr void @_ZN1a1b1c6MyFuncIiEE1TI1AE{{.*}}(
// PCH-NOHASH-NOT: {{[.]h[.][0-9]+[ (]}}

// --- Verify Sanitizer, XRay, and Profile special-case lists match unhashed canonical mangled names ---
// MyFunc<B> is in san.ignorelist (no sanitize_memory/address), MyFunc<B2> is NOT (has both),
// consumeStdStrings matched fun:*consumeStdStrings* then =sanitize (re-enabled, shares #SAN_ACTIVE),
// and GlobalVarTemplate is in global: ignorelist (no_sanitize_address):
// SANLIST: @_ZN1a1b1c17GlobalVarTemplateE{{[.]h[.][0-9]+[ ]}}= linkonce_odr global i32 7, no_sanitize_address, comdat, align 4{{$}}
// SANLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[SAN_IGNORED:[0-9]+]]
// SANLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[SAN_ACTIVE:[0-9]+]]
// SANLIST: define{{.*}} void @_ZN1a1b1c17consumeStdStringsEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[SAN_ACTIVE]]
// SANLIST: attributes #[[SAN_IGNORED]] = {
// SANLIST-NOT: sanitize_address
// SANLIST-NOT: sanitize_memory
// SANLIST-SAME: }
// SANLIST: attributes #[[SAN_ACTIVE]] = {{{.*}}sanitize_address{{.*}}sanitize_memory{{.*}}}

// XRay [always] on canonical MyFunc<B> overrides [never] fun:*MyFunc*:
// XRAYLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[XRAY_ALWAYS:[0-9]+]]
// XRAYLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[XRAY_NEVER:[0-9]+]]
// XRAYLIST-DAG: attributes #[[XRAY_ALWAYS]] = {{{.*}}"function-instrument"="xray-always"{{.*}}}
// XRAYLIST-DAG: attributes #[[XRAY_NEVER]] = {{{.*}}"function-instrument"="xray-never"{{.*}}}

// ProfileList =allow on canonical MyFunc<B> overrides fun:*MyFunc*=forbid (noprofile on MyFunc<B2>):
// PROFLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[PROF_ALLOW:[0-9]+]]
// PROFLIST: define weak_odr void @_ZN1a1b1c6MyFuncEv{{[.]h[.][0-9]+}}(ptr {{.*}}) #[[PROF_FORBID:[0-9]+]]
// PROFLIST: attributes #[[PROF_ALLOW]] = {
// PROFLIST-NOT: noprofile
// PROFLIST-SAME: }
// PROFLIST: attributes #[[PROF_FORBID]] = {{{.*}}noprofile{{.*}}}
