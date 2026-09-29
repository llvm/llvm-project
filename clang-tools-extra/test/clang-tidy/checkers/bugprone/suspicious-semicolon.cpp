// RUN: %check_clang_tidy %s bugprone-suspicious-semicolon %t

int x = 5;

void nop();

void correct1()
{
	if(x < 5) nop();
}

void correct2()
{
	if(x == 5)
		nop();
}

void correct3()
{
	if(x > 5)
	{
		nop();
	}
}

void fail1()
{
  if(x > 5); nop();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: potentially unintended semicolon [bugprone-suspicious-semicolon]
  // CHECK-FIXES: if(x > 5) nop();
}

void fail2()
{
	if(x == 5);
		nop();
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: potentially unintended semicolon [bugprone-suspicious-semicolon]
  // CHECK-FIXES: if(x == 5)
}

void fail3()
{
	if(x < 5);
	{
		nop();
	}
  // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: potentially unintended semicolon
  // CHECK-FIXES: if(x < 5)
}

void correct4()
{
  while(x % 5 == 1);
  nop();
}

void correct5()
{
	for(int i = 0; i < x; ++i)
		;
}

void fail4()
{
	for(int i = 0; i < x; ++i);
		nop();
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: potentially unintended semicolon
  // CHECK-FIXES: for(int i = 0; i < x; ++i)
}

void fail5()
{
	if(x % 5 == 1);
	  nop();
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: potentially unintended semicolon
  // CHECK-FIXES: if(x % 5 == 1)
}

void fail6() {
  int a = 0;
  if (a != 0) {
  } else if (a != 1);
    a = 2;
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: potentially unintended semicolon
  // CHECK-FIXES: } else if (a != 1)
}

void fail7() {
  if (true)
    ;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: potentially unintended semicolon
}

void correct6()
{
	do; while(false);
}

int correct7()
{
  int t_num = 0;
  char c = 'b';
  char *s = "a";
  if (s == "(" || s != "'" || c == '"') {
    t_num += 3;
    return (c == ')' && c == '\'');
  }

  return 0;
}

void correct8() {
  if (true)
    ;
  else {
  }
}
