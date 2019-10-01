int g_var = 2222;

void __fastcall foo(short arg_0, float arg_1) {
  char loc_0 = 'x';
  double loc_1 = 0.5678;
}

__declspec(align(128)) struct S {
  int a = 1234;
};

void bar(int arg_0) {
 S loc_0;
 int loc_1 = 5678;
}


int main(int argc, char *argv[]) {
  bool loc_0 = true;
  int loc_1 = 3333;

  foo(1111, 0.1234);
  bar(22);

  return 0;
}
