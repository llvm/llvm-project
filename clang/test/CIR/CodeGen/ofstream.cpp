// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#include "std-cxx.h"

namespace std {
template <class CharT> class basic_ofstream {
public:
  basic_ofstream();
  ~basic_ofstream();
  explicit basic_ofstream(const char *);
};

using ofstream = basic_ofstream<char>;

ofstream &operator<<(ofstream &, const string &);
} // namespace std

void foo(const char *path) {
  std::ofstream fout1(path);
  fout1 << path;
  std::ofstream fout2(path);
  fout2 << path;
}

// CIR: cir.func @_Z3fooPKc
// CIR: %[[V1:.*]] = cir.alloca !ty_std3A3Abasic_ofstream3Cchar3E, !cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>, ["fout1", init] {alignment = 1 : i64}
// CIR: %[[V2:.*]] = cir.alloca !ty_std3A3Abasic_ofstream3Cchar3E, !cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>, ["fout2", init] {alignment = 1 : i64}
// CIR: cir.try synthetic cleanup {
// CIR:   cir.call exception @_ZNSbIcEC1EPKcRKNS_9AllocatorE({{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E3A3AAllocator>) -> () cleanup {
// CIR:     cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V2]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR:     cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V1]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]
// CIR: cir.try synthetic cleanup {
// CIR:   %[[V10:.*]] = cir.call exception @_ZStlsRSt14basic_ofstreamIcERKSbIcE(%[[V2]], {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> !cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E> cleanup {
// CIR:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR:     cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V2]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR:     cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V1]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.store %[[V10]], {{.*}} : !cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>>
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]
// CIR: cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V2]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR: cir.call @_ZNSt14basic_ofstreamIcED1Ev(%[[V1]]) : (!cir.ptr<!ty_std3A3Abasic_ofstream3Cchar3E>) -> ()
// CIR: cir.return

// LLVM: @_Z3fooPKc(ptr {{.*}})
// LLVM:   %[[V9:.*]] = alloca %"class.std::basic_ofstream<char>", i64 1, align 1
// LLVM:   %[[V10:.*]] = alloca %"class.std::basic_ofstream<char>", i64 1, align 1
// LLVM: {{.*}}
// LLVM:   invoke void @_ZNSbIcEC1EPKcRKNS_9AllocatorE(ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// LLVM:           to label %[[B56:.*]] unwind label %[[B57:.*]]
// LLVM: [[B56]]
// LLVM:   br label {{.*}}
// LLVM: [[B57]]
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V10]])
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V9]])
// LLVM:   br label %[[B61:.*]]
// LLVM: [[B61]]
// LLVM:   resume { ptr, i32 } {{.*}}
// LLVM: {{.*}}
// LLVM:   {{.*}} = invoke ptr @_ZStlsRSt14basic_ofstreamIcERKSbIcE(ptr %[[V10]], ptr {{.*}})
// LLVM:           to label {{.*}} unwind label %[[B70:.*]]
// LLVM: [[B70]]
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V10]])
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V9]])
// LLVM:   br label %[[B74:.*]]
// LLVM: [[B74]]
// LLVM:   resume { ptr, i32 } {{.*}}
// LLVM: {{.*}}
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   br label %[[B80:.*]]
// LLVM: [[B80]]
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V10]])
// LLVM:   call void @_ZNSt14basic_ofstreamIcED1Ev(ptr %[[V9]])
// LLVM:   ret void
