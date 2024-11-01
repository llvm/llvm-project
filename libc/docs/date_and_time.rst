=======================
Date and Time Functions
=======================

.. include:: check.rst

---------------
Source location
---------------

-   The main source for time functions is located at: ``libc/src/time``

---------------------
Implementation Status
---------------------

============= =======
Function_Name C99
============= =======
clock
mktime        |check|
time
asctime       |check|
ctime
gmtime        |check|
localtime
strftime
============= =======

===================   =======
Function_Name         POSIX
===================   =======
asctime               |check|
asctime_r             |check|
clock
clock_getcpuclockid
clock_getres
clock_gettime         |check|
clock_nanosleep
clock_settime
ctime
ctime_r
difftime              |check|
getdate
gettimeofday          |check|
gmtime                |check|
gmtime_r              |check|
localtime
localtime_r
mktime                |check|
nanosleep             |check|
strftime
strptime
time
timer_create
timer_delete
timer_gettime
timer_getoverrun
timer_settime
tzset
===================   =======

