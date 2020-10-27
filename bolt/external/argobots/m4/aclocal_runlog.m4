dnl
dnl PAC_RUN_LOG mimics _AC_RUN_LOG which is autoconf internal routine.
dnl We also make sure PAC_RUN_LOG can be used in AS_IF, so the last
dnl test command should have terminating ]), i.e. without newline before ]).
dnl
AC_DEFUN([PAC_RUNLOG],[
{ AS_ECHO(["$as_me:$LINENO: $1"]) >&AS_MESSAGE_LOG_FD
  (eval $1) 2>&AS_MESSAGE_LOG_FD
  ac_status=$?
  AS_ECHO(["$as_me:$LINENO: \$? = $ac_status"]) >&AS_MESSAGE_LOG_FD
  test $ac_status = 0; }])

dnl
dnl PAC_COMMAND_IFELSE is written to replace AC_TRY_EVAL with added logging
dnl to config.log, i.e. AC_TRY_EVAL does not log anything to config.log.
dnl If autoconf provides AC_COMMAND_IFELSE or AC_EVAL_IFELSE,
dnl AC_COMMAND_IFELSE dnl should be replaced by the official autoconf macros.
dnl
dnl PAC_COMMAND_IFELSE(COMMMAND,[ACTION-IF-RUN-OK],[ACTION-IF-RUN-FAIL])
dnl
AC_DEFUN([PAC_COMMAND_IFELSE],[
AS_IF([PAC_RUNLOG([$1])],[
    $2
],[
    AS_ECHO(["$as_me: program exited with status $ac_status"]) >&AS_MESSAGE_LOG_FD
    m4_ifvaln([$3],[
        (exit $ac_status)
        $3
    ])
])
])

AC_DEFUN([PAC_RUNLOG_IFELSE],[
dnl pac_TESTLOG is the internal temporary logfile for this macro.
pac_TESTLOG="pac_test.log"
rm -f $pac_TESTLOG
PAC_COMMAND_IFELSE([$1 > $pac_TESTLOG],[
    ifelse([$2],[],[],[$2])
],[
    AS_ECHO(["*** $1 :"]) >&AS_MESSAGE_LOG_FD
    cat $pac_TESTLOG >&AS_MESSAGE_LOG_FD
    ifelse([$3],[],[],[$3])
])
rm -f $pac_TESTLOG
])


dnl PAC_VAR_PUSHVAL(VARNAME, [LastSavedValue]))
dnl
dnl Save the content of the shell variable, VARNAME, onto a stack.
dnl The saved value of VARNAME is restorable with respect to the nesting
dnl of the macro.
dnl
dnl The Last saved value of VARNAME on the stack is stored in shell variable
dnl pac_LastSavedValueOf_$VARNAME if the 2nd argument is NOT supplied.
dnl If the 2nd argument is present, the last saved value will be stored
dnl in the 2nd argument instead.
dnl
dnl The First saved value of VARNAME on the stack is stored in shell variable
dnl dnl pac_FirstSavedValueOf_$VARNAME.
dnl
AC_DEFUN([PAC_VAR_PUSHVAL],[
# START of PUSHVAL
dnl define local m4-name pac_stk_level.
AS_VAR_PUSHDEF([pac_stk_level], [pac_stk_$1_level])
AS_VAR_SET_IF([pac_stk_level],[
    AS_VAR_ARITH([pac_stk_level], [$pac_stk_level + 1])
],[
    AS_VAR_SET([pac_stk_level], [0])
])
dnl AS_ECHO_N(["PUSHVAL: pac_stk_level = $pac_stk_level, "])
dnl Save the content of VARNAME, i.e. $VARNAME, onto the stack.
AS_VAR_SET([pac_stk_$1_$pac_stk_level],[$$1])
AS_VAR_IF([pac_stk_level], [0], [
    dnl Save the 1st pushed value of VARNAME as pac_FirstSavedValueOf_$VARNAME
    AS_VAR_COPY([pac_FirstSavedValueOf_$1],[pac_stk_$1_$pac_stk_level])
])
ifelse([$2],[],[
    dnl Save the last pushed value of VARNAME as pac_LastSavedValueOf_$VARNAME
    AS_VAR_COPY([pac_LastSavedValueOf_$1],[pac_stk_$1_$pac_stk_level])
    dnl AS_ECHO(["pac_LastSavedValueOf_$1 = $pac_LastSavedValueOf_$1"])
],[
    dnl Save the last pushed value of VARNAME as $2
    AS_VAR_COPY([$2],[pac_stk_$1_$pac_stk_level])
    dnl AS_ECHO(["$2 = $$2"])
])
AS_VAR_POPDEF([pac_stk_level])
# END of PUSHVAL
])
dnl
dnl
dnl
dnl PAC_VAR_POPVAL(VARNAME)
dnl
dnl Restore variable, VARNAME, from the stack.
dnl This macro is safe with respect to the nesting.
dnl Some minimal checking of nesting balance of PAC_VAR_PUSH[POP]VAL()
dnl is done here.
dnl
AC_DEFUN([PAC_VAR_POPVAL],[
# START of POPVAL
dnl define local m4-name pac_stk_level.
AS_VAR_PUSHDEF([pac_stk_level], [pac_stk_$1_level])
AS_VAR_SET_IF([pac_stk_level],[
    AS_VAR_IF([pac_stk_level],[-1],[
        AC_MSG_WARN(["Imbalance of PUSHVAL/POPVAL of $1"])
    ],[
        dnl AS_ECHO_N(["POPVAL: pac_stk_level = $pac_stk_level, "])
        AS_VAR_COPY([$1],[pac_stk_$1_$pac_stk_level])
        dnl AS_ECHO(["popped_val = $$1"])
        AS_VAR_ARITH([pac_stk_level], [ $pac_stk_level - 1 ])
    ])
],[
    AC_MSG_WARN(["Uninitialized PUSHVAL/POPVAL of $1"])
])
AS_VAR_POPDEF([pac_stk_level])
# END of POPVAL
])
dnl
dnl
dnl
dnl PAC_COMPILE_IFELSE_LOG is a wrapper around AC_COMPILE_IFELSE with the
dnl output of ac_compile to a specified logfile instead of AS_MESSAGE_LOG_FD
dnl
dnl PAC_COMPILE_IFELSE_LOG(logfilename, input,
dnl                        [action-if-true], [action-if-false])
dnl
dnl where input, [action-if-true] and [action-if-false] are used
dnl in AC_COMPILE_IFELSE(input, [action-if-true], [action-if-false]).
dnl This macro is nesting safe.
dnl
AC_DEFUN([PAC_COMPILE_IFELSE_LOG],[
dnl
dnl Instead of defining our own ac_compile and do AC_TRY_EVAL
dnl on these variables.  We modify ac_compile used by AC_*_IFELSE
dnl by piping the output of the command to a logfile.  The reason is that
dnl 1) AC_TRY_EVAL is discouraged by Autoconf. 2) defining our ac_compile
dnl could mess up the usage and order of *CFLAGS, LDFLAGS and LIBS in
dnl these commands, i.e. deviate from how GNU standard uses these variables.
dnl
dnl Replace ">&AS_MESSAGE_LOG_FD" by "> FILE 2>&1" in ac_compile.
dnl Save a copy of ac_compile on a stack
dnl which is safe through nested invocations of this macro.
PAC_VAR_PUSHVAL([ac_compile])
dnl Modify ac_compile based on the unmodified ac_compile.
ac_compile="`echo $pac_FirstSavedValueOf_ac_compile | sed -e 's|>.*$|> $1 2>\&1|g'`"
AC_COMPILE_IFELSE([$2],[
    ifelse([$3],[],[:],[$3])
],[
    ifelse([$4],[],[:],[$4])
])
dnl Restore the original ac_compile from the stack.
PAC_VAR_POPVAL([ac_compile])
])
dnl
dnl
dnl
dnl PAC_LINK_IFELSE_LOG is a wrapper around AC_LINK_IFELSE with the
dnl output of ac_link to a specified logfile instead of AS_MESSAGE_LOG_FD
dnl
dnl PAC_LINK_IFELSE_LOG(logfilename, input,
dnl                     [action-if-true], [action-if-false])
dnl
dnl where input, [action-if-true] and [action-if-false] are used
dnl in AC_LINK_IFELSE(input, [action-if-true], [action-if-false]).
dnl This macro is nesting safe.
dnl
AC_DEFUN([PAC_LINK_IFELSE_LOG],[
dnl
dnl Instead of defining our own ac_link and do AC_TRY_EVAL
dnl on these variables.  We modify ac_link used by AC_*_IFELSE
dnl by piping the output of the command to a logfile.  The reason is that
dnl 1) AC_TRY_EVAL is discouraged by Autoconf. 2) defining our ac_link
dnl could mess up the usage and order of *CFLAGS, LDFLAGS and LIBS in
dnl these commands, i.e. deviate from how GNU standard uses these variables.
dnl
dnl Replace ">&AS_MESSAGE_LOG_FD" by "> FILE 2>&1" in ac_link.
dnl Save a copy of ac_link on a stack
dnl which is safe through nested invocations of this macro.
PAC_VAR_PUSHVAL([ac_link])
dnl Modify ac_link based on the unmodified ac_link.
ac_link="`echo $pac_FirstSavedValueOf_ac_link | sed -e 's|>.*$|> $1 2>\&1|g'`"
dnl
AC_LINK_IFELSE([$2],[
    ifelse([$3],[],[:],[$3])
],[
    ifelse([$4],[],[:],[$4])
])
dnl Restore the original ac_link from the stack.
PAC_VAR_POPVAL([ac_link])
])
dnl
dnl
dnl
dnl PAC_COMPLINK_IFELSE (input1, input2, [action-if-true], [action-if-false])
dnl
dnl where input1 and input2 are either AC_LANG_SOURCE or AC_LANG_PROGRAM
dnl enclosed input programs.
dnl
dnl The macro first compiles input1 and uses the object file created
dnl as part of LIBS during linking.  This macro is nesting safe.
dnl
AC_DEFUN([PAC_COMPLINK_IFELSE],[
AC_COMPILE_IFELSE([$1],[
    PAC_RUNLOG([mv conftest.$OBJEXT pac_conftest.$OBJEXT])
    PAC_VAR_PUSHVAL([LIBS])
    LIBS="pac_conftest.$OBJEXT $pac_FirstSavedValueOf_LIBS"
    AC_LINK_IFELSE([$2],[
        ifelse([$3],[],[:],[$3])
    ],[
        ifelse([$4],[],[:],[$4])
    ])
    PAC_VAR_POPVAL([LIBS])
    rm -f pac_conftest.$OBJEXT
],[
    ifelse([$4],[],[:],[$4])
])
])
