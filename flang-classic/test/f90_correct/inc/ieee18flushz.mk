#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee18flushz  ########

ieee18flushz: ieee18flushz.$(OBJX)
	

ieee18flushz.$(OBJX):  $(SRC)/ieee18flushz.f90
	-$(RM) ieee18flushz.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee18flushz.f90 -o ieee18flushz.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee18flushz.$(OBJX) check.$(OBJX) $(LIBS) -o ieee18flushz.$(EXESUFFIX)


ieee18flushz.run: ieee18flushz.$(OBJX)
	@echo ------------------------------------ executing test ieee18flushz
	ieee18flushz.$(EXESUFFIX)

verify: ;
build: ieee18flushz.$(OBJX) ;
run: ieee18flushz.run ;
