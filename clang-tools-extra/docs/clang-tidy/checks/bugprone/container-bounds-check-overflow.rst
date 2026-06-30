.. title:: clang-tidy - bugprone-container-bounds-check-overflow

bugprone-container-bounds-check-overflow
========================================

Finds potential overflow in unsigned integer addition before comparison with a container's
``size()`` method. It flags all of the following combinations:
- ``a + b < v.size()``
- ``a + b <= v.size()``
- ``a + b > v.size()``
- ``a + b >= v.size()``
- ``v.size() < a + b``
- ``v.size() <= a + b``
- ``v.size() > a + b``
- ``v.size() >= a + b``

The addition ``a + b`` can overflow if ``a`` and ``b`` are large enough, leading to incorrect behavior.
For example, if ``a`` is ``UINT_MAX`` and ``b`` is ``1``, then ``a + b`` will wrap around to ``0``,
and the comparison can be true, even if the container is empty.

The comparison is flagged only if size of the unsigned integers being added is the same
as the size of the container's ``size()`` return type. Smaller types are promoted to the size
of the container's ``size()`` return type before the addition, so they are safe from overflow.

Options
-------

.. option:: IgnoredContainers

    When set, the check will ignore the specified containers. The value is a
    comma-separated list of fully qualified container names. For example, to ignore
    ``std::array`` and ``CustomClass``, set the option to `::std::array,::CustomClass`.
    The default is empty string, meaning no containers are ignored.
