// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -std=c++17 %s

// Thread safety attributes on a redeclaration of a template also apply to the
// instantiations of the template that exist when the redeclaration is seen.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
};

Mutex mu;
int guarded __attribute__((guarded_by(mu)));

struct Tag {};

// The specialization exists before the attribute is seen. The template is
// defined later.
template <class T> void acquire(Mutex &m);
void *ref_acquire = (void *)&acquire<Tag>;
template <class T> void acquire(Mutex &m) __attribute__((acquire_capability(m)));
template <class T> void acquire(Mutex &m) { m.Lock(); }

void call_acquire() {
  acquire<Tag>(mu);
  guarded = 1;
  mu.Unlock();
}

// The same for a template that is not defined in this translation unit.
template <class T> void acquire_undef(Mutex &m);
void *ref_acquire_undef = (void *)&acquire_undef<Tag>;
template <class T>
void acquire_undef(Mutex &m) __attribute__((acquire_capability(m)));

void call_acquire_undef() {
  acquire_undef<Tag>(mu);
  guarded = 1;
  mu.Unlock();
}

template <class T> void need(Mutex &m);
void *ref_need = (void *)&need<Tag>;
template <class T> void need(Mutex &m) __attribute__((requires_capability(m)));

void call_need() {
  need<Tag>(mu); // expected-warning {{calling function 'need<Tag>' requires holding mutex 'mu' exclusively}}
}

// An attribute written on a definition is also applied.
template <class T> bool try_acquire(Mutex &m);
void *ref_try_acquire = (void *)&try_acquire<Tag>;
template <class T>
bool try_acquire(Mutex &m) __attribute__((try_acquire_capability(true, m))) {
  return false;
}

void call_try_acquire() {
  if (try_acquire<Tag>(mu)) {
    guarded = 1;
    mu.Unlock();
  }
}

// An explicit specialization does not get the attributes of the template.
template <class T> void release(Mutex &m);
template <> void release<Tag>(Mutex &m);
template <class T> void release(Mutex &m) __attribute__((release_capability(m)));

void call_release() {
  mu.Lock(); // expected-note {{mutex acquired here}}
  release<Tag>(mu);
} // expected-warning {{mutex 'mu' is still held at the end of function}}

// A member function template of a class.
struct Holder {
  template <class T> void acquire(Mutex &m);
};
auto ref_holder_acquire = &Holder::acquire<Tag>;
template <class T>
void Holder::acquire(Mutex &m) __attribute__((acquire_capability(m))) {
  m.Lock();
}

void call_holder_acquire(Holder &h) {
  h.acquire<Tag>(mu);
  guarded = 1;
  mu.Unlock();
}

// A member function of a class template that has been instantiated before
// the out-of-line definition of the member is seen.
template <class U> struct Wrapper {
  void acquire(Mutex &m);
  template <class T> void acquire_tmpl(Mutex &m);
};
Wrapper<int> wrapper;
auto ref_wrapper_acquire_tmpl = &Wrapper<int>::acquire_tmpl<Tag>;
template <class U>
void Wrapper<U>::acquire(Mutex &m) __attribute__((acquire_capability(m))) {
  m.Lock();
}
template <class U>
template <class T>
void Wrapper<U>::acquire_tmpl(Mutex &m) __attribute__((acquire_capability(m))) {
  m.Lock();
}

void call_wrapper_acquire() {
  wrapper.acquire(mu);
  guarded = 1;
  mu.Unlock();
}

void call_wrapper_acquire_tmpl() {
  wrapper.acquire_tmpl<Tag>(mu);
  guarded = 1;
  mu.Unlock();
  // A specialization of the member function template of the existing class
  // template instantiation that is instantiated after the definition.
  wrapper.acquire_tmpl<int>(mu);
  guarded = 2;
  mu.Unlock();
}
