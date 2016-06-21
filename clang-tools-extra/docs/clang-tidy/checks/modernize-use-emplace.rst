.. title:: clang-tidy - modernize-use-emplace

modernize-use-emplace
=====================

This check looks for cases when inserting new element into an STL
container (``std::vector``, ``std::deque``, ``std::list``) or ``llvm::SmallVector``
but the element is constructed temporarily.

Before:

.. code:: c++

	std::vector<MyClass> v;
	v.push_back(MyClass(21, 37));

	std::vector<std::pair<int,int> > w;
	w.push_back(std::make_pair(21, 37));

After:

.. code:: c++

	std::vector<MyClass> v;
	v.emplace_back(21, 37);

	std::vector<std::pair<int,int> > w;
	v.emplace_back(21, 37);

The other situation is when we pass arguments that will be converted to a type
inside a container.

Before:

.. code:: c++

	std::vector<boost::optional<std::string> > v;
	v.push_back("abc");

After:

.. code:: c++

	std::vector<boost::optional<std::string> > v;
	v.emplace_back("abc");


In this case the calls of ``push_back`` won't be replaced.

.. code:: c++
	std::vector<std::unique_ptr<int> > v;
	v.push_back(new int(5));
	auto *ptr = int;
	v.push_back(ptr);

This is because replacing it with ``emplace_back`` could cause a leak of this
pointer if ``emplace_back`` would throw exception before emplacement
(e.g. not enough memory to add new element).

For more info read item 42 - "Consider emplacement instead of insertion."
of Scott Meyers Efective Modern C++.

Check also fires if any argument of constructor call would be:
- bitfield (bitfields can't bind to rvalue/universal reference)
- ``new`` expression (to avoid leak)
or if the argument would be converted via derived-to-base cast.

This check requires C++11 of higher to run.

