class X
{
  friend class A;
};


template <typename T1, typename T2>
class B
{
};

template <class T>
struct C
{
};

class D
{
    B<D, class A> x;
    friend struct C<A>;
};
