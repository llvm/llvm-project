#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05c  ########

CWD   := $(shell pwd)
INVOKEE=runieee
FFLAGS += -Mpreprocess

ieee05c: ieee05c.$(OBJX)

ieee05c.$(OBJX):  $(SRC)/ieee05c.f90
	-$(RM) ieee05c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05c.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05c.f90 -o ieee05c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05c.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05c.$(EXESUFFIX)

ieee05c.run: ieee05c.$(OBJX)
	@echo ------------------------------------ executing test ieee05c
	$(shell ./$(INVOKEE) > ieee05c.res 2> ieee05c.err)
	@cat ieee05c.res

run: ieee05c.$(OBJX)
	@echo ------------------------------------ executing test ieee05c
	$(shell ./$(INVOKEE) > ieee05c.res 2> ieee05c.err)
	@cat ieee05c.res


build:	ieee05c.$(OBJX)
verify:	;
