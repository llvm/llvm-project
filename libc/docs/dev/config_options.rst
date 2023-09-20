.. _configure_options:

=================================
Adding new libc configure options
=================================

`There are a number of configure options <../configure.html>`_ which can be used
to configure the libc build. The config system is driven by a set of
hierarchical JSON files. At the top of the hierarchy is a JSON file by name
``config.json`` in the ``config`` directory. This JSON file lists the libc
options which affect all platforms. The default value for the option and a short
description about it listed against each option. For example:

.. code-block::

   {
     "printf": {
       "LIBC_CONF_PRINTF_DISABLE_FLOAT": {
         "value": false,
         "doc": "Disable printing floating point values in printf and friends."
       },
       ...
     }
   }

The above config indicates that the option ``LIBC_CONF_PRINTF_DISABLE_FLOAT``
has a value of ``false``. A platform, say the baremetal platform, can choose
to override this value in its ``config.json`` file in the ``config/baremetal``
directory with the following contents:

.. code-block::

   {
     "printf": {
       "LIBC_CONF_PRINTF_DISABLE_FLOAT": {
         "value": true
       }
     }
   }

Here, the config for the baremetal platform overrides the common ``false``
value of the ``LIBC_CONF_PRINTF_DISABLE_FLOAT`` with the ``true`` value.

Config JSON format
==================

Named tags
----------

As can be noted from the above examples, ``config.json`` files contains a
top-level dictionary. The keys of this dictionary are the names of
*grouping-tags*. A grouping-tag is nothing but a named tag to refer to a related
group of libc options. In the above example, a tag named ``printf`` is used to
group all libc options which affect the behavior of ``printf`` and friends.

Tag values
----------

The value corresponding to each grouping tag is also a dictionary called the
*option-dictionary*. The keys of the option-dictionary are the names of the libc
options belonging to that grouping tag. For the ``printf`` tag in the above
example, the option-dictionary is:

.. code-block::

   {
     "LIBC_CONF_PRINTF_DISABLE_FLOAT": {
       "value": false,
       "doc": 
     },
     ...
   }

The value corresponding to an option key in the option-dictionary is another
dictionary with two keys: ``"value"`` and ``"doc"``. The ``"value"`` key has
the value of the option listed against it, and the ``"doc"`` key has a short
description of the option listed against it. Note that only the main config
file ``config/config.json`` includes the ``"doc"`` key. Options which are of
``ON``/``OFF`` kind take boolean values ``true``/``false``. Other types of
options can take an integral or string value as suitable for that option. In
the above option-dictionary, the option-key ``LIBC_CONF_PRINTF_DISABLE_FLOAT``
is of boolean type with value ``true``.

Option name format
------------------

The option names, or the keys of a option-dictionary, have the following format:

.. code-block::

   LIBC_CONF_<UPPER_CASE_TAG_NAME>_<ACTION_INDICATING_THE_INTENDED_SEMANTICS>

The option name used in the above examples, ``LIBC_CONF_PRINTF_DISABLE_FLOAT``
to disable printing of floating point numbers, follows this format: It has the
prefix ``LIBC_CONF_``, followed by the grouping-tag name ``PRINTF`` in upper
case, followed by the action to disable floating point number printing
``LIBC_CONF_PRINTF_DISABLE_FLOAT``.

Mechanics of config application
===============================

Config reading
--------------

At libc config time, three different ``config.json`` files are read in the
following order:

1. ``config/config.json``
2. ``config/<platform or OS>/config.json`` if present.
3. ``config/<platform or OS>/<target arch>/config.json`` if present.

Each successive ``config.json`` file overrides the option values set by
previously read ``config.json`` files. Likewise, a similarly named command line
option to the cmake command will override the option values specified in all or
any of these ``config.json`` files. That is, users will be able to override the
config options from the command line.

Config application
------------------

Local to the directory where an option group is relevant, suitable build logic
should convert the CMake config options to appropriate compiler and/or linker
flags. Those compile/link flags can be used in listing the affected targets as
follows:

.. code-block::

   add_object_library(
    ...
    COMPILE_OPTIONS
      ${common_printf_compile_options}
      ... # Other compile options affecting this target irrespective of the
          # libc config options
   )

Note that the above scheme is only an example and not a prescription.
Developers should employ a scheme appropriate to the option being added.

Automatic doc update
====================

The CMake configure step automatically generates the user document
``doc/configure.rst``, which contains user information about the libc configure
options, using the information in the main ``config/config.json`` file.
An update to ``config/config.json`` will trigger reconfiguration by CMake, which
in turn will regenerate the documentation in ``doc/configure.rst``.
