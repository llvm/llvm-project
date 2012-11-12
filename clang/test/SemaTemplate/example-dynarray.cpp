// RUN: %clangxx -emit-llvm -c -o - %s
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

// Placement new requires <new> to be included, but we don't support that yet.
void* operator new(size_t, void* ptr) throw() {
  return ptr;
}
void operator delete(void*, void*) throw() {
}

template<typename T>
class dynarray {
public:
  dynarray() { Start = Last = End = 0; }

  dynarray(const dynarray &other) {
    Start = (T*)malloc(sizeof(T) * other.size());
    Last = End = Start + other.size();

    for (unsigned I = 0, N = other.size(); I != N; ++I)
      new (Start + I) T(other[I]);
  }
  
  ~dynarray() {
    for (unsigned I = 0, N = size(); I != N; ++I)
      Start[I].~T();
    
    free(Start);
  }

  dynarray &operator=(const dynarray &other) {
    T* NewStart = (T*)malloc(sizeof(T) * other.size());

    for (unsigned I = 0, N = other.size(); I != N; ++I)
      new (NewStart + I) T(other[I]);

    for (unsigned I = 0, N = size(); I != N; ++I)
      Start[I].~T();
    
    free(Start);
    Start = NewStart;
    Last = End = NewStart + other.size();
    return *this;
  }

  unsigned size() const { return Last - Start; }
  unsigned capacity() const { return End - Start; }

  void push_back(const T& value);
  
  void pop_back() {
    --Last;
    Last->~T();
  }

  T& operator[](unsigned Idx) {
    return Start[Idx];
  }

  const T& operator[](unsigned Idx) const {
    return Start[Idx];
  }

  typedef T* iterator;
  typedef const T* const_iterator;

  iterator begin() { return Start; }
  const_iterator begin() const { return Start; }
  
  iterator end() { return Last; }
  const_iterator end() const { return Last; }
  
  bool operator==(const dynarray &other) const {
    if (size() != other.size())
      return false;
    
    for (unsigned I = 0, N = size(); I != N; ++I)
      if ((*this)[I] != other[I])
        return false;
    
    return true;
  }
  
  bool operator!=(const dynarray &other) const {
    return !(*this == other);
  }
  
public:
  T* Start, *Last, *End;
};

template<typename T>
void dynarray<T>::push_back(const T& value) {
  if (Last == End) {
    unsigned NewCapacity = capacity() * 2;
    if (NewCapacity == 0)
      NewCapacity = 4;
    
    T* NewStart = (T*)malloc(sizeof(T) * NewCapacity);
    
    unsigned Size = size();
    for (unsigned I = 0; I != Size; ++I)
      new (NewStart + I) T(Start[I]);
    
    for (unsigned I = 0, N = size(); I != N; ++I)
      Start[I].~T();
    free(Start);
    
    Start = NewStart;
    Last = Start + Size;
    End = Start + NewCapacity;
  }
  
  new (Last) T(value);
  ++Last;
}

struct Point { 
  Point() { x = y = z = 0.0; }
  Point(const Point& other) : x(other.x), y(other.y), z(other.z) { }

  float x, y, z;
};

int main() {
  dynarray<int> di;
  di.push_back(0);
  di.push_back(1);
  di.push_back(2);
  di.push_back(3);
  di.push_back(4);
  assert(di.size() == 5);
  for (dynarray<int>::iterator I = di.begin(), IEnd = di.end(); I != IEnd; ++I)
    assert(*I == I - di.begin());

  for (int I = 0, N = di.size(); I != N; ++I)
    assert(di[I] == I);

  di.pop_back();
  assert(di.size() == 4);
  di.push_back(4);

  dynarray<int> di2 = di;
  assert(di2.size() == 5);
  assert(di.begin() != di2.begin());
  for (dynarray<int>::iterator I = di2.begin(), IEnd = di2.end(); 
       I != IEnd; ++I)
    assert(*I == I - di2.begin());

  dynarray<int> di3(di);
  assert(di3.size() == 5);
  assert(di.begin() != di3.begin());
  for (dynarray<int>::iterator I = di3.begin(), IEnd = di3.end(); 
       I != IEnd; ++I)
    assert(*I == I - di3.begin());

  dynarray<int> di4;
  assert(di4.size() == 0);
  di4 = di;
  assert(di4.size() == 5);
  assert(di.begin() != di4.begin());
  for (dynarray<int>::iterator I = di4.begin(), IEnd = di4.end(); 
       I != IEnd; ++I)
    assert(*I == I - di4.begin());

  assert(di4 == di);
  di4[3] = 17;
  assert(di4 != di);
  
  dynarray<Point> dp;
  dp.push_back(Point());
  assert(dp.size() == 1);
  
  return 0;
}
