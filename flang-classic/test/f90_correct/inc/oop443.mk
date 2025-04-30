#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop443  ########


oop443: run
	

build:  $(SRC)/oop443.f90
	-$(RM) oop443.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop443.f90 -o oop443.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop443.$(OBJX) check.$(OBJX) $(LIBS) -o oop443.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop443
	oop443.$(EXESUFFIX)

verify: ;

