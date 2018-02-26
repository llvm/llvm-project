// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha -analyzer-max-loop 2 -analyzer-config widen-loops=true -analyzer-output=text -verify %s

int *p_a;
int bar();
int flag_a;
int test_for_bug_25609() {
  if (p_a == 0) // expected-note {{Assuming 'p_a' is equal to null}} 
                // expected-note@-1 {{Taking true branch}}
    bar();
  for (int i = 0;  // expected-note {{Loop condition is true.  Entering loop body}}                    
                   // expected-note@-1 {{Loop condition is false. Execution continues on line 16}}
       ++i,        // expected-note {{Value assigned to 'p_a'}} 
       i < flag_a;
       ++i) {}
                                      
  *p_a = 25609; // no-crash expected-warning {{Dereference of null pointer (loaded from variable 'p_a')}}
                // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p_a')}}
  return *p_a;
}

int flag_b;
int while_analyzer_output() {
  flag_b = 100;
  int num = 10;
  while (flag_b-- > 0) { // expected-note {{Loop condition is true.  Entering loop body}} 
                         // expected-note@-1 {{Value assigned to 'num'}} 
                         // expected-note@-2 {{Loop condition is false. Execution continues on line 30}}
    num = flag_b;
  }
  if (num < 0) // expected-note {{Assuming 'num' is >= 0}} 
               // expected-note@-1 {{Taking false branch}}
    flag_b = 0;
  else if (num >= 1) // expected-note {{Assuming 'num' is < 1}} 
                     // expected-note@-1 {{Taking false branch}}
    flag_b = 50;
  else
    flag_b = 100;
  return flag_b / num; // no-crash expected-warning {{Division by zero}} 
                       // expected-note@-1 {{Division by zero}}
}

int flag_c;
int do_while_analyzer_output() {
  int num = 10;
  do {   // expected-note {{Loop condition is true. Execution continues on line 47}} 
         // expected-note@-1 {{Loop condition is false.  Exiting loop}}
    num--;
  } while (flag_c-- > 0); //expected-note {{Value assigned to 'num'}}
  int local = 0;
  if (num == 0)       // expected-note {{Assuming 'num' is equal to 0}} 
                      // expected-note@-1 {{Taking true branch}}
    local = 10 / num; // no-crash expected-warning {{Division by zero}}
                      // expected-note@-1 {{Division by zero}}
  return local;
}

int flag_d;
int test_for_loop() {
  int num = 10;
  for (int i = 0;    // expected-note {{Loop condition is true.  Entering loop body}} 
                     // expected-note@-1 {{Loop condition is false. Execution continues on line 67}}
       new int(10),  // expected-note {{Value assigned to 'num'}}
       i < flag_d;
       ++i) {         
    ++num;
  }
  if (num == 0) // expected-note {{Assuming 'num' is equal to 0}} 
                // expected-note@-1 {{Taking true branch}}
    flag_d += 10;
  return flag_d / num; // no-crash expected-warning {{Division by zero}} 
                       // expected-note@-1 {{Division by zero}}
}
