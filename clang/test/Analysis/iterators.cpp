// RUN: %clang --analyze -Xclang -analyzer-checker=core,experimental.cplusplus.Iterators -Xclang -verify %s
// XFAIL: win32

#include <vector>

void fum(std::vector<int>::iterator t);

void foo1()
{
  // iterators that are defined but not initialized
  std::vector<int>::iterator it2;
  fum(it2); // expected-warning{{Use of iterator that is not defined}}
  *it2;     // expected-warning{{Use of iterator that is not defined}}

  std::vector<int> v, vv;
  std::vector<int>::iterator it = v.begin();
  fum(it);  // no-warning
  *it;  // no-warning
  // a valid iterator plus an integer is still valid
  std::vector<int>::iterator et = it + 3;
  while(it != et) { // no-warning
    if (*it == 0) // no-warning
      *it = 1;  // no-warning
  }
  // iterators from different instances Cannot be compared
  et = vv.end();
  while(it != et) // expected-warning{{Cannot compare iterators from different containers}}
    ;

  for( std::vector<int>::iterator it = v.begin(); it != v.end(); it++ ) { // no-warning
    if (*it == 1) // no-warning
      *it = 0;  // no-warning
  }

  // copying a valid iterator results in a valid iterator
  et = it;  // no-warning
  *et;  // no-warning

  // any combo of valid iterator plus a constant is still valid
  et = it + 2;  // no-warning
  *et;  // no-warning
  et = 2 + it;  // no-warning
  *et;  // no-warning
  et = 2 + 4 + it;  // no-warning
  *et;  // no-warning

  // calling insert invalidates unless assigned to as result, but still
  // invalidates other iterators on the same instance
  it = v.insert( it, 1 ); // no-warning
  *et;  // expected-warning{{Attempt to use an iterator made invalid by call to 'insert'}}
  ++it; // no-warning

  // calling erase invalidates the iterator
  v.erase(it);  // no-warning
  et = it + 2;  // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  et = 2 + it + 2;  // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  et = 2 + it;  // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  ++it; // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  it++; // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  *it;  // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  it = v.insert( it, 1 ); // expected-warning{{Attempt to use an iterator made invalid by call to 'erase'}}
  // now valid after return from insert
  *it;  // no-warning
}

// work with using namespace
void foo2()
{
  using namespace std;

  vector<int> v;
  vector<int>::iterator it = v.begin();
  *it;  // no-warning
  v.insert( it, 1 );  // no-warning
  *it;  // expected-warning{{Attempt to use an iterator made invalid by call to 'insert'}}
  it = v.insert( it, 1 ); // expected-warning{{Attempt to use an iterator made invalid by call to 'insert'}}
  *it;  // no-warning
}

// using reserve eliminates some warnings
void foo3()
{
  std::vector<long> v;
  std::vector<long>::iterator b = v.begin();
  v.reserve( 100 );

  // iterator assigned before the reserve is still invalidated
  *b; // expected-warning{{Attempt to use an iterator made invalid by call to 'reserve'}}
  b = v.begin();
  v.insert( b, 1 ); // no-warning

  // iterator after assignment is still valid (probably)
  *b; // no-warning
}

// check on copying one iterator to another
void foo4()
{
  std::vector<float> v, vv;
  std::vector<float>::iterator it = v.begin();
  *it;  // no-warning
  v = vv;
  *it;  // expected-warning{{Attempt to use an iterator made invalid by copying another container to its container}}
}

