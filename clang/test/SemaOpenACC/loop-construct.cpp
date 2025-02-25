// RUN: %clang_cc1 %s -fopenacc -verify -Wno-empty-body -Wno-unused-value
namespace std {
  struct random_access_iterator_tag{};
}

struct SomeStruct{
  void operator++();
  void operator++(int);
};

struct SomeIterator {
  bool operator!=(SomeIterator&);
  void operator++();
  void operator++(int);
  int operator*();
};

struct SomeRAIterator {
  using iterator_category = std::random_access_iterator_tag;
  SomeRAIterator();
  SomeRAIterator(int i);

  void operator=(int i);
  SomeRAIterator &operator=(SomeRAIterator&);

  void operator++(int);
  void operator++();
  int operator*();
  void operator+=(int);
  bool operator!=(SomeRAIterator&);
};

struct HasIteratorCollection {
  SomeIterator &begin();
  SomeIterator &end();
};

struct HasRAIteratorCollection {
  SomeRAIterator &begin();
  SomeRAIterator &end();
};

void func_call();

template<typename Int, typename IntPtr, typename Float, typename Struct, typename Iterator, typename RandAccessIterator>
void SeqLoopRules() {

  // No rules in this section!
#pragma acc loop seq
  for(;;);

#pragma acc loop seq
  for(float f;;);

#pragma acc loop seq
  for(int f;;);

#pragma acc loop seq
  for(int f,g;;);

#pragma acc loop seq
  for(Int f;;);

#pragma acc loop seq
  for(Int *f;;);

#pragma acc loop seq
  for(IntPtr f;;);

#pragma acc loop seq
  for(Float *f;;);

#pragma acc loop seq
  for(SomeStruct f;;);

#pragma acc loop seq
  for(Struct f;;);

#pragma acc loop seq
  for(SomeIterator f;;);

#pragma acc loop seq
  for(Iterator f;;);

#pragma acc loop seq
  for(SomeRAIterator f;;);

#pragma acc loop seq
  for(RandAccessIterator f;;);

#pragma acc loop seq
  for(Int f;;);

#pragma acc loop seq
  for(Int f;;++f);

#pragma acc loop seq
  for(Int f;;f+=1);

  int i;
#pragma acc loop seq
  for(Int f;;i+=1);

#pragma acc loop seq
  for(Int f;;i++);

#pragma acc loop seq
  for(RandAccessIterator f;;i++);

#pragma acc loop seq
  for(RandAccessIterator f;;func_call());

  Int Array[5];
#pragma acc loop seq
  for(auto X : Array);

#pragma acc loop seq
  for(auto X : HasIteratorCollection{});

#pragma acc loop seq
  for(auto X : HasRAIteratorCollection{});

  RandAccessIterator f;
#pragma acc loop seq
  for(f;;f++);

#pragma acc loop seq
  for(f = 0;;++f);

#pragma acc loop seq
  for(f = 0;;f++);

#pragma acc loop seq
  for(f = 0;;f+=1);
}


template<typename Int, typename IntPtr, typename Float, typename Struct, typename Iterator, typename RandAccessIterator>
void LoopRules() {

  // Loop variable must be integer, pointer, or random_access_iterator
#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(;;);

#pragma acc loop
  // expected-error@+6{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'float')}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(float f = 0;;);

#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(int f;;);

#pragma acc loop
  // expected-error@+6 2{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2 2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(int f,g;;);

#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(Int f;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(Int *f = nullptr;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(IntPtr f = nullptr;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(Float *f = nullptr;;++f);

#pragma acc loop
  // expected-error@+6{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'SomeStruct')}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(SomeStruct f;;);

#pragma acc loop
  // expected-error@+4{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'SomeStruct')}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(Struct f;;++f);

#pragma acc loop
  // expected-error@+4{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'SomeIterator')}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(SomeIterator f;;++f);

#pragma acc loop
  // expected-error@+4{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'SomeIterator')}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(Iterator f;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(SomeRAIterator f;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(RandAccessIterator f;;++f);

  Int i;
#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-4{{'loop' construct is here}}
  for( i = 0;;);

#pragma acc loop
  // expected-error@+6 2{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2 2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for( i;;);

#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for( int j ;;);

#pragma acc loop
  // expected-error@+6 2{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2 2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for( int j, k = 0;;);

#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(Int f;;);

#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(Int f;;++f);

#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  for(Int f;;f+=1);

#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(Int f;;i+=1);

#pragma acc loop
  // expected-error@+6{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-4{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-6{{'loop' construct is here}}
  for(Int f;;i++);

#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-4{{'loop' construct is here}}
  for(RandAccessIterator f;;i++);

#pragma acc loop
  // expected-error@+4{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  // expected-error@+2{{OpenACC 'loop' variable must monotonically increase or decrease ('++', '--', or compound assignment)}}
  // expected-note@-4{{'loop' construct is here}}
  for(RandAccessIterator f;;func_call());

  // Not much we can do here other than check for random access iterator.
  Int Array[5];
#pragma acc loop
  for(auto X : Array);

#pragma acc loop
  // expected-error@+2 2{{loop variable of loop associated with an OpenACC 'loop' construct must be of integer, pointer, or random-access-iterator type (is 'SomeIterator')}}
  // expected-note@-2 2{{'loop' construct is here}}
  for(auto X : HasIteratorCollection{});

#pragma acc loop
  for(auto X : HasRAIteratorCollection{});

  RandAccessIterator end;
#pragma acc loop
  for(RandAccessIterator f = 0; f != end; ++f);

  RandAccessIterator f;
#pragma acc loop
  // expected-error@+2 2{{OpenACC 'loop' construct must have initialization clause in canonical form ('var = init' or 'T var = init'}}
  // expected-note@-2 2{{'loop' construct is here}}
  for(f;f != end;f++);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(f = 0;;++f);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(f = 0;;f++);

#pragma acc loop
  // expected-error@+2{{OpenACC 'loop' construct must have a terminating condition}}
  // expected-note@-2{{'loop' construct is here}}
  for(f = 0;;f+=1);

#pragma acc loop
  for(f = 0;f != end;++f);

#pragma acc loop
  for(f = 0;f != end;f++);

#pragma acc loop
  for(f = 0;f != end;f+=1);
}

void inst() {
  SeqLoopRules<int, int*, float, SomeStruct, SomeIterator, SomeRAIterator>();
  // expected-note@+1{{in instantiation of function template specialization}}
  LoopRules<int, int*, float, SomeStruct, SomeIterator, SomeRAIterator>();
}

