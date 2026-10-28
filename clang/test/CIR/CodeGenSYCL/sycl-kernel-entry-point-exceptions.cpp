// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fcxx-exceptions -fexceptions -fsycl-is-host -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fcxx-exceptions -fexceptions -fsycl-is-host -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fcxx-exceptions -fexceptions -fsycl-is-host -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Validate generation of exception handling code for functions declared
// with the sycl_kernel_entry_point attribute that implicitly call a
// sycl_kernel_launch function that may throw an exception. Exception
// handling is not relevant for the generated offload kernel entry point
// function, so device compilation is intentionally not exercised.

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// A generic kernel object type.
template<int, int = 0>
struct KT {
  void operator()() const;
};


// Validate that exception handling instructions are omitted when a
// potentially throwing sycl_kernel_entry_point attributed function
// calls a potentially throwing sycl_kernel_launch function (a thrown
// exception will propagate with no explicit handling required).
namespace ns1 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<1>)]]
  void skep(KT<1> k) {
    k();
  }
}
// CIR: cir.func {{.*}}@_ZN3ns14skepE2KTILi1ELi0EE() attributes {{{.*}}mustprogress, optnone} {
// CIR-NOT: cir.try
// CIR: cir.call @_ZN3ns118sycl_kernel_launchI2KNILi1EEJ2KTILi1ELi0EEEEEvPKcDpT0_(%{{.*}}) : (
// CIR: cir.return

// LLVM: ; Function Attrs: mustprogress noinline optnone
// LLVM-NEXT: define dso_local void @_ZN3ns14skepE2KTILi1ELi0EE() #{{[0-9]+}} {
// LLVM-NOT: invoke
// LLVM: call void @_ZN3ns118sycl_kernel_launchI2KNILi1EEJ2KTILi1ELi0EEEEEvPKcDpT0_(ptr noundef @.str)
// LLVM: ret void


// Validate that exception handling instructions are emitted when a
// non-throwing sycl_kernel_entry_point attributed function calls
// a potentially throwing sycl_kernel_launch function.
namespace ns2 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<2>)]]
  void skep(KT<2> k) noexcept {
    k();
  }
}
// CIR:      cir.func {{.*}}@_ZN3ns24skepE2KTILi2ELi0EE() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
// CIR:        cir.try {
// CIR:          %[[STR:.*]] = cir.get_global @".str.1" : !cir.ptr<!cir.array<!s8i x 14>>
// CIR-NEXT:     %[[STR_PTR:.*]] = cir.cast array_to_ptrdecay %[[STR]] : !cir.ptr<!cir.array<!s8i x 14>> -> !cir.ptr<!s8i>
// CIR:          cir.call @_ZN3ns218sycl_kernel_launchI2KNILi2EEJ2KTILi2ELi0EEEEEvPKcDpT0_(%[[STR_PTR]]) : (!cir.ptr<!s8i> {llvm.noundef}) -> ()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } catch all (%[[TOK:[a-z0-9]+]]: !cir.eh_token
// CIR-NEXT:     cir.eh.terminate %[[TOK]] : !cir.eh_token
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM:      ; Function Attrs: mustprogress noinline nounwind optnone
// LLVM-NEXT: define dso_local void @_ZN3ns24skepE2KTILi2ELi0EE() #{{[0-9]+}} personality ptr @__gxx_personality_v0 {
// LLVM:        invoke void @_ZN3ns218sycl_kernel_launchI2KNILi2EEJ2KTILi2ELi0EEEEEvPKcDpT0_(ptr noundef @.str.1)
// LLVM-NEXT:     to label %[[CONT:[a-z0-9.]+]] unwind label %[[LPAD:[a-z0-9.]+]]
// LLVM-EMPTY:
// LLVM-NEXT: {{^}}[[CONT]]:
// OGCG-NEXT:   ret void
// LLVM-CIR-NEXT: br label %[[RET:[0-9]+]]
// LLVM-EMPTY:
// LLVM-NEXT: {{^}}[[LPAD]]:
// LLVM-NEXT:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM-NEXT:     catch ptr null
// LLVM-NEXT:   %[[EXN:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG-NEXT:   call void @__clang_call_terminate(ptr %[[EXN]]) #[[TERM_ATTR:[0-9]+]]
// OGCG-NEXT:   unreachable
// OGCG-NEXT: }

// ClangIR forwards the exception to the terminate call through two extra
// single-predecessor blocks and places the normal return block last.
// LLVM-CIR-NEXT:   %[[SEL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM-CIR-NEXT:   br label %[[BB1:[0-9]+]]
// LLVM-CIR-EMPTY:
// LLVM-CIR-NEXT: {{^}}[[BB1]]:
// LLVM-CIR-NEXT:   %[[EXN1:.*]] = phi ptr [ %[[EXN]], %[[LPAD]] ]
// LLVM-CIR-NEXT:   %[[SEL1:.*]] = phi i32 [ %[[SEL]], %[[LPAD]] ]
// LLVM-CIR-NEXT:   br label %[[BB2:[0-9]+]]
// LLVM-CIR-EMPTY:
// LLVM-CIR-NEXT: {{^}}[[BB2]]:
// LLVM-CIR-NEXT:   %[[EXN2:.*]] = phi ptr [ %[[EXN1]], %[[BB1]] ]
// LLVM-CIR-NEXT:   %{{.*}} = phi i32 [ %[[SEL1]], %[[BB1]] ]
// LLVM-CIR-NEXT:   call void @__clang_call_terminate(ptr %[[EXN2]]) #[[TERM_ATTR:[0-9]+]]
// LLVM-CIR-NEXT:   unreachable
// LLVM-CIR-EMPTY:
// LLVM-CIR-NEXT: {{^}}[[RET]]:
// LLVM-CIR-NEXT:   ret void
// LLVM-CIR-NEXT: }


// Validate that exception handling instructions are omitted when a
// potentially throwing sycl_kernel_entry_point attributed function
// calls a non-throwing sycl_kernel_launch function (a thrown
// exception will terminate within sycl_kernel_launch).
namespace ns3 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) noexcept;
  [[clang::sycl_kernel_entry_point(KN<3>)]]
  void skep(KT<3> k) {
    k();
  }
}
// CIR: cir.func {{.*}}@_ZN3ns34skepE2KTILi3ELi0EE() attributes {{{.*}}mustprogress, optnone} {
// CIR-NOT: cir.try
// CIR: cir.call @_ZN3ns318sycl_kernel_launchI2KNILi3EEJ2KTILi3ELi0EEEEEvPKcDpT0_(%{{.*}}) nothrow : (
// CIR: cir.return

// ClangIR does not yet infer nounwind for functions that contain no
// potentially throwing calls.
// LLVM-CIR: ; Function Attrs: mustprogress noinline optnone
// OGCG: ; Function Attrs: mustprogress noinline nounwind optnone
// LLVM-NEXT: define dso_local void @_ZN3ns34skepE2KTILi3ELi0EE() #{{[0-9]+}} {
// LLVM-NOT: invoke
// LLVM: call void @_ZN3ns318sycl_kernel_launchI2KNILi3EEJ2KTILi3ELi0EEEEEvPKcDpT0_(ptr noundef @.str.2) #{{[0-9]+}}
// LLVM: ret void


// Validate that exception handling instructions are omitted when a
// non-throwing sycl_kernel_entry_point attributed function calls a
// non-throwing sycl_kernel_launch function.
namespace ns4 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) noexcept;
  [[clang::sycl_kernel_entry_point(KN<4>)]]
  void skep(KT<4> k) noexcept {
    k();
  }
}
// ClangIR wraps the body of a noexcept function in a terminate scope even
// when the only call it contains is nothrow.
// CIR: cir.func {{.*}}@_ZN3ns44skepE2KTILi4ELi0EE() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
// CIR: cir.try {
// CIR: cir.call @_ZN3ns418sycl_kernel_launchI2KNILi4EEJ2KTILi4ELi0EEEEEvPKcDpT0_(%{{.*}}) nothrow : (
// CIR: cir.eh.terminate

// The terminate scope lowers to no EH instructions, but ClangIR still attaches
// a personality function.
// LLVM: ; Function Attrs: mustprogress noinline nounwind optnone
// LLVM-CIR-NEXT: define dso_local void @_ZN3ns44skepE2KTILi4ELi0EE() #{{[0-9]+}} personality ptr @__gxx_personality_v0 {
// OGCG-NEXT: define dso_local void @_ZN3ns44skepE2KTILi4ELi0EE() #{{[0-9]+}} {
// LLVM-NOT: invoke
// LLVM: call void @_ZN3ns418sycl_kernel_launchI2KNILi4EEJ2KTILi4ELi0EEEEEvPKcDpT0_(ptr noundef @.str.3) #{{[0-9]+}}
// LLVM: ret void

// LLVM: attributes #[[TERM_ATTR]] = { noreturn nounwind }
