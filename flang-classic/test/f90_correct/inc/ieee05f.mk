#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05f  ########

CWD   := $(shell pwd)
INVOKEE=runieee

ieee05f: ieee05f.$(OBJX)

ieee05f.$(OBJX):  $(SRC)/ieee05f.f90
	-$(RM) ieee05f.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05f.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05f.f90 -o ieee05f.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05f.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05f.$(EXESUFFIX)


ieee05f.run: ieee05f.$(OBJX)
	@echo ------------------------------------ executing test ieee05f
	$(shell ./$(INVOKEE) > ieee05f.res 2> ieee05f.err)
	@cat ieee05f.res
run: ieee05f.$(OBJX)
	@echo ------------------------------------ executing test ieee05f
	$(shell ./$(INVOKEE) > ieee05f.res 2> ieee05f.err)
	@cat ieee05f.res

build:	ieee05f.$(OBJX)
verify:	;

