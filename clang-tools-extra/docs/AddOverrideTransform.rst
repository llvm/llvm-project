.. index:: Add-Override Transform

======================
Add-Override Transform
======================

The Add-Override Transform adds the ``override`` specifier to member
functions that override a virtual function in a base class and that
don't already have the specifier. The transform is enabled with the 
:option:`-add-override` option of :program:`cpp11-migrate`.
For example:

.. code-block:: c++

  class A {
  public:
    virtual void h() const;
  };

  class B : public A {
  public:
    void h() const;

    // The declaration of h is transformed to
    void h() const override;
  };


Known Limitations
-----------------
* This transform will fail if a method declaration has an inlined method
  body and there is a comment between the method declaration and the body.
  In this case, the override keyword will incorrectly be inserted at the 
  end of the comment.

.. code-block:: c++

  class B : public A {
  public:
    virtual void h() const // comment
    { }

    // The declaration of h is transformed to
    virtual void h() const // comment override
    { }
  };

