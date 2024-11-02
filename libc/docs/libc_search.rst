=============
Search Tables
=============

.. include:: check.rst

---------------
Source Location
---------------

-   The main source for search functions is located at:
    ``libc/src/search``.

-   Hashtable implementation is located at:
    ``libc/src/__support/HashTable``.

-   The tests are located at:
    ``libc/test/src/search/``.

---------------------
Implementation Status
---------------------

POSIX Standard Types
====================

============================ =========
Type Name                    Available
============================ =========
ACTION                       |check|
ENTRY                        |check|
VISIT                        
============================ =========

POSIX Standard Functions
========================

============================ =========
Function Name                Available
============================ =========
hcreate                      |check|
hdestroy                     |check|
hsearch                      |check|
insque                       |check|
lfind                        
lsearch                      
remque                       |check|
tdelete
tfind
tsearch
twalk
============================ =========


GNU Extension Functions
=======================

=========================  =========
Function Name              Available
=========================  =========
hsearch_r                  |check|
hcreate_r                  |check|
hdestroy_r                 |check|
tdestroy
twalk_r
=========================  =========


Standards
=========
search.h is specified in POSIX.1-200x (Portable Operating System Interface, Volume1: Base Specifications).
