// RUN: env CINDEXTEST_KEEP_GOING=1 c-index-test -code-completion-at=%s:25:1 %s
// Shouldn't crash!
// This is the minimized test that triggered an infinite recursion:

+(BOOL) onEntity {
}

-(const Object &) a_200 {
}

-(int) struct {
}

-(int) bar {
}

-(int) part {
}

+(some_type_t) piece {
}

+(void) z_Z_42 {
  ([self onEntity: [] { 42];
  } class: ^ {  }
];
  [super];
  BOOL struct;
}
