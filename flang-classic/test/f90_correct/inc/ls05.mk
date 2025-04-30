#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls05  ########


ls05: run
	

build:  $(SRC)/ls05.f
	-$(RM) ls05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls05.f -o ls05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls05.$(OBJX) check.$(OBJX) $(LIBS) -o ls05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls05
	ls05.$(EXESUFFIX)

verify: ;

ls05.run: run

