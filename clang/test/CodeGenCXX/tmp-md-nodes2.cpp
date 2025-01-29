// REQUIRES: asserts
// RUN: %clang_cc1 -O0 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -O0 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -mllvm --experimental-debuginfo-iterators=true %s -o - | \
// RUN: FileCheck %s

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
