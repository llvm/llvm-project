// RUN: %clang_cc1 -fsyntax-only -verify %s

// this exercises the CUDA tokenization where >>> is treated
// as a single token, ensuring that the <<<< and >>>> sequences from
// conflict markers are not incorrectly interpreted as C++/CUDA syntax. 

// expected-error@+1 {{version control conflict marker in file}}
<<<<<<< HEAD 
int x = 5;
=======
int y = 0;
>>>>>>> other-branch
