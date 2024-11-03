// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only %s

struct X { 
  int array[0]; 
  int array1[0]; 
  int array2[0]; 
  X();
  ~X();
};

struct Y {
  int first;
  X padding;
  int second;
};

int zero_size_array[(sizeof(Y)  == 8) -1]; // no error here!
