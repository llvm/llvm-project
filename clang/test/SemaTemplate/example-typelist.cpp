// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// A simple cons-style typelist
struct nil { };

template<typename Head, typename Tail = nil>
struct cons { 
  typedef Head head;
  typedef Tail tail;
};

// is_same trait, for testing
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

// metaprogram that computes the length of a list
template<typename T> struct length;

template<typename Head, typename Tail>
struct length<cons<Head, Tail> > {
  static const unsigned value = length<Tail>::value + 1;
};

template<>
struct length<nil> {
  static const unsigned value = 0;
};

typedef cons<unsigned char, 
             cons<unsigned short, 
                  cons<unsigned int,
                       cons<unsigned long> > > > unsigned_inttypes;
int length0[length<unsigned_inttypes>::value == 4? 1 : -1];

// metaprogram that reverses a list

// FIXME: I would prefer that this be a partial specialization, but
// that requires partial ordering of class template partial
// specializations.
template<typename T> 
class reverse {
  typedef typename reverse<typename T::tail>::type reversed_tail;

  typedef typename reverse<typename reversed_tail::tail>::type most_of_tail;

public:
  typedef cons<typename reversed_tail::head,
               typename reverse<cons<typename T::head, most_of_tail> >::type> type;
};

template<typename Head>
class reverse<cons<Head> > {
public:
  typedef cons<Head> type;
};

template<>
class reverse<nil> {
public:
  typedef nil type;
};

int reverse0[is_same<reverse<unsigned_inttypes>::type,
                     cons<unsigned long, 
                          cons<unsigned int, 
                               cons<unsigned short,
                                    cons<unsigned char> > > > >::value? 1 : -1];

// metaprogram that finds a type within a list

// FIXME: I would prefer that this be a partial specialization, but
// that requires partial ordering of class template partial
// specializations.
template<typename List, typename T>
struct find : find<typename List::tail, T> { };

template<typename Tail, typename T>
struct find<cons<T, Tail>, T> {
  typedef cons<T, Tail> type;
};

template<typename T>
struct find<nil, T> {
  typedef nil type;
};

int find0[is_same<find<unsigned_inttypes, unsigned int>::type,
                       cons<unsigned int, cons<unsigned long> > >::value?
             1 : -1];
int find1[is_same<find<unsigned_inttypes, int>::type, nil>::value? 1 : -1];

