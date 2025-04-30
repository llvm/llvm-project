#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc30  ########


fc30: run
	

build:  $(SRC)/fc30.f
	-$(RM) fc30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc30.f -o fc30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc30.$(OBJX) check.$(OBJX) $(LIBS) -o fc30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc30
	fc30.$(EXESUFFIX)

verify: ;

fc30.run: run

