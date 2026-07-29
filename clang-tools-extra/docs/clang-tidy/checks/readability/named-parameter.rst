.. title:: clang-tidy - readability-named-parameter

readability-named-parameter
===========================

Find functions with unnamed arguments.

The check implements the following rule originating in the Google C++ Style
Guide:

https://google.github.io/styleguide/cppguide.html#Function_Declarations_and_Definitions

All parameters should have the same name in both the function declaration and
definition. If a parameter is not utilized, its name can be commented out in a
function definition.

.. code-block:: c++

    int doingSomething(int a, int b, int c);

    int doingSomething(int a, int b, int /*c*/) {
        // Ok: the third param is not used
        return a + b;
    }

Corresponding cpplint.py check name: `readability/function`.

The check ignores parameters whose types are standard tag types (e.g.
``std::in_place_t``, ``std::allocator_arg_t``, ``std::nothrow_t``,
iterator tags, lock tags, etc.). The set of ignored types can be customized
with the :option:`IgnoredTypes` option.

Options
-------

.. option:: InsertPlainNamesInForwardDecls

   If set to `true`, the check will insert parameter names without comments for
   forward declarations only. Otherwise, the check will insert parameter names
   as comments (e.g., ``/*param*/``). Default is `false`.

.. option:: IgnoredTypes

   A semicolon-separated list of fully-qualified type names whose parameters
   do not need to be named (for example, tag dispatch types, iterator tags,
   etc.). Defaults to the standard tag types:

   .. code-block:: text

      std::adopt_lock_t
      std::allocator_arg_t
      std::bidirectional_iterator_tag
      std::contiguous_iterator_tag
      std::default_sentinel_t
      std::defer_lock_t
      std::destroying_delete_t
      std::forward_iterator_tag
      std::from_range_t
      std::in_place_index_t
      std::in_place_t
      std::in_place_type_t
      std::input_iterator_tag
      std::nothrow_t
      std::nostopstate_t
      std::nullopt_t
      std::output_iterator_tag
      std::piecewise_construct_t
      std::random_access_iterator_tag
      std::sorted_equivalent_t
      std::sorted_unique_t
      std::try_to_lock_t
      std::unexpect_t
      std::unreachable_sentinel_t
