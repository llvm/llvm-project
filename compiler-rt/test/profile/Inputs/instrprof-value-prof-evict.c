void callee_0() {}
void callee_1() {}
void callee_2() {}
void callee_3() {}

void *CalleeAddrs[] = {callee_0, callee_1, callee_2, callee_3};
extern void lprofSetMaxValsPerSite(unsigned);

// sequences of callee ids

// In the following sequences,
// there are two targets, the dominating target is
// target 0.
int CallSeqTwoTarget_1[] = {0, 0, 0, 0, 0, 1, 1};
int CallSeqTwoTarget_2[] = {1, 1, 0, 0, 0, 0, 0};
int CallSeqTwoTarget_3[] = {1, 0, 0, 1, 0, 0, 0};
int CallSeqTwoTarget_4[] = {0, 0, 0, 1, 0, 1, 0};

// In the following sequences, there are three targets
// The dominating target is 0 and has > 50% of total
// counts.
int CallSeqThreeTarget_1[] = {0, 0, 0, 0, 0, 0, 1, 2, 1};
int CallSeqThreeTarget_2[] = {1, 2, 1, 0, 0, 0, 0, 0, 0};
int CallSeqThreeTarget_3[] = {1, 0, 0, 2, 0, 0, 0, 1, 0};
int CallSeqThreeTarget_4[] = {0, 0, 0, 1, 0, 1, 0, 0, 2};

// Four target sequence --
// There are two cold targets which occupies the value counters
// early. There is also a very hot target and a medium hot target
// which are invoked in an interleaved fashion -- the length of each
// hot period in the sequence is shorter than the cold targets' count.
//  1. If only two values are tracked, the Hot and Medium hot targets
//     should surive in the end
//  2. If only three values are tracked, the top three targets should
//     surive in the end.
int CallSeqFourTarget_1[] = {1, 1, 1, 2, 2, 2, 2, 0, 0, 3, 0, 0, 3, 0, 0, 3,
                             0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3};

// Same as above, but the cold entries are invoked later.
int CallSeqFourTarget_2[] = {0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0,
                             0, 3, 0, 0, 3, 0, 0, 3, 1, 1, 1, 2, 2, 2, 2};

// Same as above, but all the targets are interleaved.
int CallSeqFourTarget_3[] = {0, 3, 0, 0, 1, 3, 0, 0, 0, 2, 0, 0, 3, 3, 0, 3,
                             2, 2, 0, 3, 3, 1, 0, 0, 1, 0, 0, 3, 0, 2, 0};

typedef void (*FPT)(void);


// Testing value profiling eviction algorithm.
FPT getCalleeFunc(int I) { return CalleeAddrs[I]; }

int main() {
  int I;

#define INDIRECT_CALLSITE(Sequence, NumValsTracked)                            \
  lprofSetMaxValsPerSite(NumValsTracked);                                      \
  for (I = 0; I < sizeof(Sequence) / sizeof(*Sequence); I++) {                 \
    FPT FP = getCalleeFunc(Sequence[I]);                                       \
    FP();                                                                      \
  }

  // check site, target patterns
  // CHECK: 0, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqTwoTarget_1, 1);

  // CHECK-NEXT: 1, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqTwoTarget_2, 1);

  // CHECK-NEXT: 2, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqTwoTarget_3, 1);

  // CHECK-NEXT: 3, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqTwoTarget_4, 1);

  // CHECK-NEXT: 4, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqThreeTarget_1, 1);

  // CHECK-NEXT: 5, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqThreeTarget_2, 1);

  // CHECK-NEXT: 6, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqThreeTarget_3, 1);

  // CHECK-NEXT: 7, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqThreeTarget_4, 1);

  // CHECK-NEXT: 8, {{_?}}callee_0
  // CHECK-NEXT: 8, {{_?}}callee_1
  INDIRECT_CALLSITE(CallSeqThreeTarget_1, 2);

  // CHECK-NEXT: 9, {{_?}}callee_0
  // CHECK-NEXT: 9, {{_?}}callee_1
  INDIRECT_CALLSITE(CallSeqThreeTarget_2, 2);

  // CHECK-NEXT: 10, {{_?}}callee_0
  // CHECK-NEXT: 10, {{_?}}callee_1
  INDIRECT_CALLSITE(CallSeqThreeTarget_3, 2);

  // CHECK-NEXT: 11, {{_?}}callee_0
  // CHECK-NEXT: 11, {{_?}}callee_1
  INDIRECT_CALLSITE(CallSeqThreeTarget_4, 2);

  // CHECK-NEXT: 12, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqFourTarget_1, 1);

  // CHECK-NEXT: 13, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqFourTarget_2, 1);

  // CHECK-NEXT: 14, {{_?}}callee_0
  INDIRECT_CALLSITE(CallSeqFourTarget_3, 1);

  // CHECK-NEXT: 15, {{_?}}callee_0
  // CHECK-NEXT: 15, {{_?}}callee_3
  INDIRECT_CALLSITE(CallSeqFourTarget_1, 2);

  // CHECK-NEXT: 16, {{_?}}callee_0
  // CHECK-NEXT: 16, {{_?}}callee_3
  INDIRECT_CALLSITE(CallSeqFourTarget_2, 2);

  // CHECK-NEXT: 17, {{_?}}callee_0
  // CHECK-NEXT: 17, {{_?}}callee_3
  INDIRECT_CALLSITE(CallSeqFourTarget_3, 2);

  // CHECK-NEXT: 18, {{_?}}callee_0
  // CHECK-NEXT: 18, {{_?}}callee_3
  // CHECK-NEXT: 18, {{_?}}callee_2
  INDIRECT_CALLSITE(CallSeqFourTarget_1, 3);

  // CHECK-NEXT: 19, {{_?}}callee_0
  // CHECK-NEXT: 19, {{_?}}callee_3
  // CHECK-NEXT: 19, {{_?}}callee_2
  INDIRECT_CALLSITE(CallSeqFourTarget_2, 3);

  // CHECK-NEXT: 20, {{_?}}callee_0
  // CHECK-NEXT: 20, {{_?}}callee_3
  // CHECK-NEXT: 20, {{_?}}callee_2
  INDIRECT_CALLSITE(CallSeqFourTarget_3, 3);

  return 0;
}
