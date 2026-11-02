// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR,BEFORE,BEFORE-NOOPT
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O2 -fno-unroll-loops -falign-loops=32 -emit-cir %s -o %t-opts.cir
// RUN: FileCheck --input-file=%t-opts.cir %s --check-prefix=CIR,BEFORE,BEFORE-OPT
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR,AFTER

void use(int);
int arr[3];

// CIR-LABEL: cir.func {{.*}} @_Z17must_progress_fori
void must_progress_for(int n) {
  for (int i = 0; i < n; ++i)
    use(i);
  // BEFORE: cir.for : cond {
  // BEFORE: } body {
  // BEFORE: } step {
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[INIT:.*]]{{$}}
  // AFTER: ^[[INIT]]:
  // AFTER-NEXT: %[[I:.*]] = cir.alloca "i"
  // AFTER-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // AFTER-NEXT: cir.store align(4) %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
  // AFTER-NEXT: cir.br ^[[COND:.*]]

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: %[[LOAD_I:.*]] = cir.load align(4) %
  // AFTER-NEXT: %[[LOAD_N:.*]] = cir.load align(4) %
  // AFTER-NEXT: %[[CMP:.*]] = cir.cmp lt %[[LOAD_I]], %[[LOAD_N]] : !s32i
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[BODY:.*]], ^[[END:.*]]{{$}}

  // AFTER: ^[[BODY]]:
  // AFTER-NEXT: cir.load align(4)
  // AFTER-NEXT: cir.call @_Z3usei
  // AFTER-NEXT: cir.br ^[[STEP:.*]]{{$}}

  // AFTER: ^[[STEP]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.inc
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: cir.br ^[[COND]] loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z19must_progress_whilei
void must_progress_while(int n) {
  while (n--)
    use(n);

  // BEFORE: cir.while {
  // BEFORE: } do {
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START:.*]]{{$}}

  // AFTER: ^[[START]]:
  // AFTER-NEXT: cir.br ^[[COND:.*]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.dec
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: %[[CMP:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[BODY:.*]], ^[[END:.*]]{{$}}

  // AFTER: ^[[BODY]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei
  // AFTER-NEXT: cir.br ^[[COND]] loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return

}

// CIR-LABEL: cir.func {{.*}} @_Z22must_progress_do_whilei
void must_progress_do_while(int n) {
  do {
    use(n);
  } while (n > 0);
  // BEFORE: cir.do {
  // BEFORE: } while {
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START:.*]]{{$}}

  // AFTER: ^[[START]]:
  // AFTER-NEXT:  cir.br ^[[BODY_LANDING_PAD:.*]]{{$}}

  // AFTER: ^[[COND:.*]]:  // pred:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.const #cir.int<0> : !s32i
  // AFTER-NEXT: %[[CMP:.*]] = cir.cmp
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[NEXT_ITR:.*]], ^[[END:.*]] loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // AFTER: ^[[NEXT_ITR]]:
  // AFTER-NEXT: cir.br ^[[BODY:.*]]{{$}}

  // AFTER: ^[[BODY]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei(
  // AFTER-NEXT: cir.br ^[[RETURN_TO_COND:.*]]{{$}}

  // AFTER: ^[[RETURN_TO_COND]]:
  // AFTER-NEXT: cir.br ^[[COND]]{{$}}

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z23must_progress_range_fori
void must_progress_range_for(int n) {
  for (int x : arr)
    use(x);
  // Note: NO loop hints on range for, inherited behavior from OGCG.
  // BEFORE: cir.scope {
  // BEFORE:   cir.for : cond {
  // BEFORE:   } body {
  // BEFORE:   } step {
  // BEFORE:   } loc
  // BEFORE: } loc
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START:.*]]{{$}}

  // AFTER: ^[[START]]:
  // Lots of stuff for iterator handling omitted.
  // AFTER: cir.br ^[[COND:.*]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: %[[CMP:.*]] = cir.cmp ne
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[BODY:.*]], ^[[END:.*]]{{$}}

  // AFTER: ^[[BODY]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei(
  // AFTER-NEXT: cir.br ^[[INC:.*]]{{$}}

  // AFTER: ^[[INC]]:
  // AFTER:   cir.load
  // AFTER:   cir.const #cir.int<1> : !s32i
  // AFTER:   cir.ptr_stride
  // AFTER:   cir.store
  // AFTER:   cir.br ^[[COND]]

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z21must_progress_inf_forv
void must_progress_inf_for() {
  for (;;) {
  }
  // NOT must-progress because it is an inf loop.
  // BEFORE: cir.scope {
  // BEFORE:   cir.for : cond {
  // BEFORE:   } body {
  // BEFORE:   } step {
  // BEFORE-NOOPT:   } loc
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = disable code_align = 32>
  // BEFORE: } loc
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START:.*]]{{$}}
  // AFTER: ^[[START]]:  // pred: ^bb0
  // AFTER-NEXT: cir.br ^[[COND:.*]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: %[[TRUE:.*]] = cir.const #true
  // AFTER-NEXT: cir.brcond %[[TRUE]] ^[[BODY:.*]], ^[[END:.*]]

  // AFTER: ^[[BODY]]:
  // AFTER-NEXT:   cir.br ^[[INC:.*]]{{$}}
  // AFTER: ^[[INC]]:
  // AFTER-NEXT:   cir.br ^[[COND]]{{$}}

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z21continue_break_in_fori
void continue_break_in_for(int n) {
  for (int i = 0; i < n; ++i) {
    if (i & 1)
      continue;
    if (i > 8)
      break;
    use(i);
  }
  // BEFORE: cir.for : cond {
  // BEFORE: } body {
  // BEFORE: } step {
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[INIT:.*]]{{$}}
  // AFTER: ^[[INIT]]:
  // AFTER-NEXT: cir.alloca "i"
  // AFTER-NEXT: cir.const #cir.int<0> : !s32i
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: cir.br ^[[COND]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: %[[CMP:.*]] = cir.cmp lt
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[NEXT:.*]], ^[[END:.*]]{{$}}

  // AFTER: ^[[NEXT]]:
  // AFTER-NEXT: cir.br ^[[EXTRA_BLOCK_1:.*]]{{$}}

  // AFTER: ^[[EXTRA_BLOCK_1]]:
  // AFTER-NEXT: cir.br ^[[IF_1_COND:.*]]{{$}}

  // AFTER: ^[[IF_1_COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.const #cir.int<1> : !s32i
  // AFTER-NEXT: cir.and
  // AFTER-NEXT: %[[AND_ONE:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[AND_ONE]] ^[[CONTINUE:.*]], ^[[BODY_AFTER_CONTINUE:.*]]{{$}}

  // AFTER: ^[[CONTINUE]]:
  // AFTER-NEXT: cir.br ^[[INC:.*]]{{$}}

  // AFTER: ^[[BODY_AFTER_CONTINUE]]:
  // AFTER-NEXT: cir.br ^[[EXTRA_BLOCK_2:.*]]{{$}}

  // AFTER: ^[[EXTRA_BLOCK_2]]:
  // AFTER-NEXT: cir.br ^[[IF_2_COND:.*]]{{$}}

  // AFTER: ^[[IF_2_COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.const #cir.int<8> : !s32i
  // AFTER-NEXT: %[[CMP:.*]] = cir.cmp gt
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[BREAK:.*]], ^[[SKIP_BREAK:.*]]{{$}}

  // AFTER: ^[[BREAK]]:
  // AFTER-NEXT: cir.br ^[[END]]

  // AFTER: ^[[SKIP_BREAK]]:
  // AFTER-NEXT:   cir.br ^[[BODY_REST:.*]]{{$}}

  // AFTER: ^[[BODY_REST]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei
  // AFTER-NEXT: cir.br ^[[BODY_END:.*]]{{$}}

  // AFTER: ^[[BODY_END]]:
  // AFTER-NEXT: cir.br ^[[INC]]{{$}}

  // AFTER: ^[[INC]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.inc
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: cir.br ^[[COND]] loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z17continue_in_whilei
void continue_in_while(int n) {
  while (n--) {
    if (n & 1)
      continue;
    use(n);
  }
  // BEFORE: cir.while {
  // BEFORE: } do {
  // BEFORE:   cir.scope {
  // BEFORE:     cir.scope {
  // BEFORE:       cir.if {{.*}} {
  // BEFORE:       } loc
  // BEFORE:     } loc
  // BEFORE:   } loc
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START:.*]]{{$}}
  // AFTER: ^[[START]]:
  // AFTER-NEXT:cir.br ^[[COND]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.dec nsw
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: %[[CMP:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[CMP]] ^[[START_BODY:.*]], ^[[END:.*]]{{$}}

  // AFTER: ^[[START_BODY]]:
  // AFTER-NEXT: cir.br ^[[START_IF:.*]]{{$}}

  // AFTER: ^[[START_IF]]:
  // AFTER-NEXT: cir.br ^[[IF_COND:.*]]{{$}}

  // AFTER: ^[[IF_COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.const #cir.int<1> : !s32i
  // AFTER-NEXT: cir.and
  // AFTER-NEXT: %[[AND:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[AND]] ^[[CONTINUE:.*]], ^[[NO_CONTINUE:.*]]{{$}}

  // AFTER: ^[[CONTINUE]]:
  // AFTER-NEXT: cir.br ^[[COND]] loop_id distinct{{\[}}[[LOOP_ID:[0-9]+]]{{\]}}<> loop_hints < must_progress>

  // AFTER: ^[[NO_CONTINUE]]:
  // AFTER-NEXT: cir.br ^[[BODY_REST:.*]]{{$}}

  // AFTER: ^[[BODY_REST]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei
  // AFTER-NEXT: cir.br ^[[NEXT_ITR:.*]]{{$}}

  // AFTER: ^[[NEXT_ITR]]:
  // AFTER-NEXT: cir.br ^[[COND]] loop_id distinct{{\[}}[[LOOP_ID]]{{\]}}<> loop_hints < must_progress>

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}

// CIR-LABEL: cir.func {{.*}} @_Z20continue_in_do_whilei
void continue_in_do_while(int n) {
  do {
    if (n & 1)
      continue;
    use(n);
  } while (--n);

  // BEFORE: cir.do {
  // BEFORE: } while {
  // BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // BEFORE-OPT:} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
  // BEFORE: cir.return

  // AFTER: cir.br ^[[START]]

  // AFTER: ^[[START]]:
  // AFTER-NEXT: cir.br ^[[BODY_START:.*]]{{$}}

  // AFTER: ^[[COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.dec nsw
  // AFTER-NEXT: cir.store
  // AFTER-NEXT: %[[AFTER_DEC:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[AFTER_DEC]] ^[[BODY_START:.*]], ^[[END:.*]] loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // AFTER: ^[[BODY_START]]:
  // AFTER-NEXT: cir.br ^[[BODY_START_2:.*]]{{$}}
 
  // AFTER: ^[[BODY_START_2]]:
  // AFTER-NEXT: cir.br ^[[IF_COND:.*]]{{$}}

  // AFTER: ^[[IF_COND]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.const #cir.int<1> : !s32i
  // AFTER-NEXT: cir.and
  // AFTER-NEXT: %[[AND:.*]] = cir.cast int_to_bool
  // AFTER-NEXT: cir.brcond %[[AND]] ^[[CONTINUE:.*]], ^[[NO_CONTINUE:.*]]{{$}}

  // AFTER: ^[[CONTINUE]]:
  // AFTER-NEXT: cir.br ^[[COND]]{{$}}

  // AFTER:  ^[[NO_CONTINUE]]:
  // AFTER-NEXT:  cir.br ^[[BODY_REST:.*]]{{$}}

  // AFTER: ^[[BODY_REST]]:
  // AFTER-NEXT: cir.load
  // AFTER-NEXT: cir.call @_Z3usei
  // AFTER-NEXT: cir.br ^[[GOTO_COND:.*]]{{$}}

  // AFTER: ^[[GOTO_COND]]:
  // AFTER-NEXT: cir.br ^[[COND]]{{$}}

  // AFTER: ^[[END]]:
  // Note: There is a dead-branch here happening too, omitted for brevity.
  // AFTER: cir.return
}


// CIR-LABEL: cir.func {{.*}} @_Z7pragmasi
void pragmas(int n) {
#pragma clang loop unroll_count(4) vectorize(enable) interleave_count(2)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4 code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4>
#pragma nounroll
  while (n)
    --n;
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable>
#pragma unroll
  do {
    --n;
  } while (n > 0);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = enable>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = enable code_align = 32>
// AFTER: cir.brcond %{{.*}} ^{{.*}}, ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = enable>
#pragma unroll 8
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 8>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 8 code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 8>
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = full>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = full code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = full>
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1 unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1>
#pragma clang loop vectorize_width(8)
  for (int x : arr)
    use(x);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_width = 8 vectorize_scalable = false>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_width = 8 vectorize_scalable = false unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_width = 8 vectorize_scalable = false>
#pragma clang loop vectorize_width(4, scalable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = true>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = true unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = true>
#pragma clang loop vectorize_width(4, fixed)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false>
#pragma clang loop vectorize(assume_safety)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable parallel>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable unroll = disable parallel code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable parallel>
#pragma clang loop vectorize_predicate(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = true>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = true unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = true>
#pragma clang loop vectorize_predicate(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = false>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = false unroll = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_predicate = false>
#pragma clang loop distribute(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress distribute = true>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable distribute = true code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress distribute = true>
#pragma clang loop distribute(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress distribute = false>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable distribute = false code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress distribute = false>
#pragma unroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam = enable>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable unroll_and_jam = enable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam = enable>
#pragma nounroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam = disable>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable unroll_and_jam = disable code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam = disable>
#pragma unroll_and_jam(4)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam_count = 4>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable unroll_and_jam_count = 4 code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_and_jam_count = 4>
#pragma clang loop pipeline(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress pipeline_disabled>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable pipeline_disabled code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress pipeline_disabled>
#pragma clang loop pipeline_initiation_interval(3)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress pipeline_initiation_interval = 3>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable pipeline_initiation_interval = 3 code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress pipeline_initiation_interval = 3>
#pragma clang loop licm(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress licm_disabled>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable licm_disabled code_align = 32>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress licm_disabled>
  [[clang::code_align(16)]]
  for (int i = 0; i < n; ++i)
    use(i);
// BEFORE-NOOPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress code_align = 16>
// BEFORE-OPT: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable code_align = 16>
// AFTER: cir.br ^{{.*}} loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress code_align = 16>
}

