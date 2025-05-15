.. title:: clang-tidy - bugprone-function-visibility-change

bugprone-function-visibility-change
===================================

Check changes in visibility of C++ member functions in subclasses. The check
detects if a virtual function is overridden with a different visibility than in
the base class declaration. Only normal functions are detected, no constructors,
operators, conversions or other special functions.

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

The changed visibility can be an indicator of bad design or a result of
coding error or code changes. If it is intentional, it can be avoided by
adding an additional virtual function with the new access.


