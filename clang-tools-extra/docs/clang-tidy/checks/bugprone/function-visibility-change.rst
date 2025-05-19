.. title:: clang-tidy - bugprone-function-visibility-change

bugprone-function-visibility-change
===================================

Checks changes in visibility of C++ member functions in subclasses. This
includes for example if a virtual function declared as `private` is overridden
and declared as `public` in a subclass. The detected change is the modification
of visibility resulting from keywords `public`, `protected`, `private` at
overridden virtual functions. Use of the `using` keyword is not considered by
this check. The check applies to any normal virtual function and optionally to
destructors or operators.

.. code-block:: c++

  class A {
  public:
    virtual void f_pub();
  private:
    virtual void f_priv();
  };
  
  class B: public A {
  public:
    void f_priv(); // warning: changed visibility from private to public
  private:
    void f_pub(); // warning: changed visibility from public to private
  };

  class C: private A {
    // no warning: f_pub becomes private in this case but this is from the
    // private inheritance
  };

  class D: private A {
  public:
    void f_pub(); // warning: changed visibility from private to public
                  // 'f_pub' would have private access but is forced to be
                  // public
  };

If the visibility is changed in this way, it can indicate bad design or
programming error. If the change is necessary, it can be achieved by the
``using`` keyword in a more safe way (this has no effect on the visibility
in further subclasses).
