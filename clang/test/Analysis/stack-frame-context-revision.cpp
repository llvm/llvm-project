// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core,cplusplus.NewDelete -verify %s

// expected-no-diagnostics:
// From now the profile of the 'StackFrameContext' also contains the
// 'NodeBuilderContext::blockCount()'. With this addition we can distinguish
// between the 'StackArgumentsSpaceRegion' of the 'P' arguments being different
// on every iteration.

typedef __INTPTR_TYPE__ intptr_t;

template <typename PointerTy>
struct SmarterPointer {
  PointerTy getFromVoidPointer(void *P) const {
    return static_cast<PointerTy>(P);
  }

  PointerTy getPointer() const {
    return getFromVoidPointer(reinterpret_cast<void *>(Value));
  }

  intptr_t Value = 13;
};

struct Node {
  SmarterPointer<Node *> Pred;
};

void test(Node *N) {
  while (N) {
    SmarterPointer<Node *> Next = N->Pred;
    delete N;

    N = Next.getPointer();
    // no-warning: 'Use of memory after it is freed' was here as the same
    //             'StackArgumentsSpaceRegion' purged out twice as 'P'.
  }
}
