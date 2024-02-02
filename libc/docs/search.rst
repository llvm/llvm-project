=============
Search Tables
=============

.. include:: check.rst

---------------
Source location
---------------

-   The main source for bitwise utility functions is located at:
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


GNU Extension Types
===================

============================ ================= =========
Type Name                    Associated Macro  Available
============================ ================= ========= 
struct qelem                 
__compar_fn_t                __COMPAR_FN_T
comparison_fn_t
__action_fn_t                __ACTION_FN_T
__free_fn_t
============================ ================= =========


POSIX Standard Functions
========================

============================ =========
Function Name                Available
============================ =========
hcreate                      |check|
hdestroy                     |check|
hsearch                      |check|
insque                       
lfind                        
lsearch                      
remque
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
twalk_r
=========================  =========


Standards
=========
search.h is specified in POSIX.1-200x (Portable Operating System Interface, Volume1: Base Specifications).
