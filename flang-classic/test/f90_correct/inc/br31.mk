#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test br31  ########


br31: run
	

build:  $(SRC)/br31.f
	-$(RM) br31.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/br31.f -o br31.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) br31.$(OBJX) check.$(OBJX) $(LIBS) -o br31.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test br31
	br31.$(EXESUFFIX)

verify: ;

br31.run: run

