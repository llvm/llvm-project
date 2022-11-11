.. title:: clang-tidy - objc-nsdate-formatter

objc-nsdate-formatter
=====================

When ``NSDateFormatter`` is used to convert an ``NSDate`` type to a ``String`` type, the user
can specify a custom format string. Certain format specifiers are undesirable
despite being legal. See http://www.unicode.org/reports/tr35/tr35-dates.html#Date_Format_Patterns for all legal date patterns.

This checker reports as warnings the following string patterns in a date format specifier:

#. yyyy + ww : Calendar year specified with week of a week year (unless YYYY is also specified).
  
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `yyyy-ww`; 
     | Output string: `2014-01` (Wrong because it’s not the first week of 2014)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `dd-MM-yyyy (ww-YYYY)`; 
     | Output string: `29-12-2014 (01-2015)` (This is correct)
    
#. F without ee/EE : Numeric day of week in a month without actual day.
    
   * | **Example:** Input Date: `29 December 2014` ; Format String: `F-MM`; 
     | Output string: `5-12` (Wrong because it reads as *5th ___ of Dec* in English)
    
#. F without MM : Numeric day of week in a month without month.
   
   * | **Example:** Input Date: `29 December 2014` ; Format String: `F-EE`
     | Output string: `5-Mon` (Wrong because it reads as *5th Mon of ___* in English)
    
#. WW without MM : Week of the month without the month.
   
   * | **Example:** Input Date: `29 December 2014` ; Format String: `WW-yyyy`
     | Output string: `05-2014` (Wrong because it reads as *5th Week of ___* in English)
    
#. YYYY + QQ : Week year specified with quarter of normal year (unless yyyy is also specified).
   
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `YYYY-QQ`
     | Output string: `2015-04` (Wrong because it’s not the 4th quarter of 2015)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `ww-YYYY (QQ-yyyy)`
     | Output string: `01-2015 (04-2014)` (This is correct)
    
#. YYYY + MM :  Week year specified with Month of a calendar year (unless yyyy is also specified).
    
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `YYYY-MM`
     | Output string: `2015-12` (Wrong because it’s not the 12th month of 2015)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `ww-YYYY (MM-yyyy)`
     | Output string: `01-2015 (12-2014)` (This is correct)
    
#. YYYY + DD : Week year with day of a calendar year (unless yyyy is also specified).
    
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `YYYY-DD`
     | Output string: `2015-363` (Wrong because it’s not the 363rd day of 2015)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `ww-YYYY (DD-yyyy)`
     | Output string: `01-2015 (363-2014)` (This is correct)
    
#. YYYY + WW : Week year with week of a calendar year (unless yyyy is also specified).
    
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `YYYY-WW`
     | Output string: `2015-05` (Wrong because it’s not the 5th week of 2015)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `ww-YYYY (WW-MM-yyyy)`
     | Output string: `01-2015 (05-12-2014)` (This is correct)
    
#. YYYY + F : Week year with day of week in a calendar month (unless yyyy is also specified).
    
   * | **Example 1:** Input Date: `29 December 2014` ; Format String: `YYYY-ww-F-EE`
     | Output string: `2015-01-5-Mon` (Wrong because it’s not the 5th Monday of January in 2015)
    
   * | **Example 2:** Input Date: `29 December 2014` ; Format String: `ww-YYYY (F-EE-MM-yyyy)`
     | Output string: `01-2015 (5-Mon-12-2014)` (This is correct)
