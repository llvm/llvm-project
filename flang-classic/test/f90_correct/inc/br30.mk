#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test br30  ########


br30: run
	

build:  $(SRC)/br30.f
	-$(RM) br30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/br30.f -o br30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) br30.$(OBJX) check.$(OBJX) $(LIBS) -o br30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test br30
	br30.$(EXESUFFIX)

verify: ;

br30.run: run

