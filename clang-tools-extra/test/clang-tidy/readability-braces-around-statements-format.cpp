// RUN: %check_clang_tidy %s readability-braces-around-statements %t -- -format-style="{IndentWidth: 3}" --

void do_something(const char *) {}

bool cond(const char *) {
  return false;
}

void test() {
  if (cond("if0") /*comment*/) do_something("same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: statement should be inside braces
  // CHECK-FIXES: {{^}}   if (cond("if0") /*comment*/) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}      do_something("same-line");{{$}}
  // CHECK-FIXES-NEXT: {{^}}   }{{$}}

  if (1) while (2) if (3) for (;;) do ; while(false) /**/;/**/
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:26: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-4]]:35: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-5]]:38: warning: statement should be inside braces
  // CHECK-FIXES:      {{^}}   if (1) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}      while (2) {
  // CHECK-FIXES-NEXT: {{^}}         if (3) {
  // CHECK-FIXES-NEXT: {{^}}            for (;;) {
  // CHECK-FIXES-NEXT: {{^}}               do {
  // CHECK-FIXES-NEXT: {{^}}                  ;
  // CHECK-FIXES-NEXT: {{^}}               } while (false) /**/; /**/
  // CHECK-FIXES-NEXT: {{^}}            }
  // CHECK-FIXES-NEXT: {{^}}         }
  // CHECK-FIXES-NEXT: {{^}}      }
  // CHECK-FIXES-NEXT: {{^}}   }
}
