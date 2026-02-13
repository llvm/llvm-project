// REQUIRES: asserts
// RUN: %clang_cc1 -O0 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// Trigger GenerateVarArgsThunk.
// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// Check that retainedNodes are properly maintained at function cloning.
// RUN: %clang_cc1 -O1 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-DI

// This test simply checks that the varargs thunk is created. The failing test
// case asserts.

typedef signed char __int8_t;
typedef int BOOL;
class CMsgAgent;

class CFs {
public:
  typedef enum {} CACHE_HINT;
  virtual BOOL ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... ) ;
};

typedef struct {} _Lldiv_t;

class CBdVfs {
public:
  virtual ~CBdVfs( ) {}
};

class CBdVfsImpl : public CBdVfs, public CFs {
  BOOL ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... );
};

BOOL CBdVfsImpl::ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... ) {
  return true;
}

// CHECK: define {{.*}} @_ZThn{{[48]}}_N10CBdVfsImpl12ReqCacheHintEP9CMsgAgentN3CFs10CACHE_HINTEz(

// An empty retainedNodes list of cloned DISubprogram.
// CHECK-DI: [[EMPTY:![0-9]+]] = !{}
// CHECK-DI: distinct !DISubprogram({{.*}}, linkageName: "_ZN10CBdVfsImpl12ReqCacheHintEP9CMsgAgentN3CFs10CACHE_HINTEz", {{.*}}, retainedNodes: [[RN1:![0-9]+]]
// A non-empty retainedNodes list of original DISubprogram.
// CHECK-DI: [[RN1]] = !{!{{.*}}}

// CHECK-DI: distinct !DISubprogram({{.*}}, linkageName: "_ZN10CBdVfsImpl12ReqCacheHintEP9CMsgAgentN3CFs10CACHE_HINTEz", {{.*}}, retainedNodes: [[EMPTY]]
