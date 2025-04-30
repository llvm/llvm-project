#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee19flushz  ########

ieee19flushz: ieee19flushz.$(OBJX)
	

ieee19flushz.$(OBJX):  $(SRC)/ieee19flushz.f90
	-$(RM) ieee19flushz.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee19flushz.f90 -o ieee19flushz.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee19flushz.$(OBJX) check.$(OBJX) $(LIBS) -o ieee19flushz.$(EXESUFFIX)


ieee19flushz.run: ieee19flushz.$(OBJX)
	@echo ------------------------------------ executing test ieee19flushz
	ieee19flushz.$(EXESUFFIX)

verify: ;
build: ieee19flushz.$(OBJX) ;
run: ieee19flushz.run ;
