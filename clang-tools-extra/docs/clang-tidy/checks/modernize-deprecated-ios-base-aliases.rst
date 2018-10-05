.. title:: clang-tidy - modernize-deprecated-ios-base-aliases

modernize-deprecated-ios-base-aliases
=====================================

This check warns the uses of the deprecated member types of ``std::ios_base``
and replaces those that have a non-deprecated equivalent.

===================================  ===========================
Deprecated member type               Replacement
===================================  ===========================
``std::ios_base::io_state``          ``std::ios_base::iostate``
``std::ios_base::open_mode``         ``std::ios_base::openmode``
``std::ios_base::seek_dir``          ``std::ios_base::seekdir``
``std::ios_base::streamoff``          
``std::ios_base::streampos``         
===================================  ===========================
