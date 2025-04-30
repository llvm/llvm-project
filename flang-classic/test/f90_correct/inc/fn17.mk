#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fn17  ########


fn17: fn17.$(OBJX)
	

fn17.$(OBJX):  $(SRC)/fn17.f90
	-$(RM) fn17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fn17.f90 -o fn17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fn17.$(OBJX) check.$(OBJX) $(LIBS) -o fn17.$(EXESUFFIX)


fn17.run: fn17.$(OBJX)
	@echo ------------------------------------ executing test fn17
	fn17.$(EXESUFFIX)
run: fn17.$(OBJX)
	@echo ------------------------------------ executing test fn17
	fn17.$(EXESUFFIX)

verify:	;
build:	fn17.$(OBJX)
