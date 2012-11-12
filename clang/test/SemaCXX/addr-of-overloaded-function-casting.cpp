// RUN: %clang_cc1 -fsyntax-only -verify %s
void g();

void f(); // expected-note 9{{candidate function}}
void f(int); // expected-note 9{{candidate function}}

template<class T> void t(T); // expected-note 6{{candidate function}}
template<class T> void t(T*); // expected-note 6{{candidate function}}

template<class T> void u(T);

int main()
{
  { bool b = (void (&)(char))f; } // expected-error{{does not match required type}}
  { bool b = (void (*)(char))f; } // expected-error{{does not match required type}}
  
  { bool b = (void (&)(int))f; } //ok
  { bool b = (void (*)(int))f; } //ok
  
  { bool b = static_cast<void (&)(char)>(f); } // expected-error{{does not match}}
  { bool b = static_cast<void (*)(char)>(f); } // expected-error{{address of overloaded function}}
  
  { bool b = static_cast<void (&)(int)>(f); } //ok
  { bool b = static_cast<void (*)(int)>(f); } //ok
  
  
  { bool b = reinterpret_cast<void (&)(char)>(f); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(char)>(f); } // expected-error{{cannot resolve}}
  
  { bool b = reinterpret_cast<void (*)(char)>(g); } //ok
  { bool b = static_cast<void (*)(char)>(g); } // expected-error{{not allowed}}
  
  { bool b = reinterpret_cast<void (&)(int)>(f); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(int)>(f); } // expected-error{{cannot resolve}}

  { bool b = (int (&)(char))t; } // expected-error{{does not match}}
  { bool b = (int (*)(char))t; } // expected-error{{does not match}}
  
  { bool b = (void (&)(int))t; } //ok
  { bool b = (void (*)(int))t; } //ok
  
  { bool b = static_cast<void (&)(char)>(t); } //ok
  { bool b = static_cast<void (*)(char)>(t); } //ok
  
  { bool b = static_cast<void (&)(int)>(t); } //ok
  { bool b = static_cast<void (*)(int)>(t); } //ok
  
  
  { bool b = reinterpret_cast<void (&)(char)>(t); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(char)>(t); } // expected-error{{cannot resolve}}
  
  { bool b = reinterpret_cast<int (*)(char)>(g); } //ok
  { bool b = static_cast<int (*)(char)>(t); } // expected-error{{cannot be static_cast}}
  { bool b = static_cast<int (&)(char)>(t); } // expected-error{{does not match required}}
  
  { bool b = static_cast<void (&)(char)>(f); } // expected-error{{does not match}}
}
