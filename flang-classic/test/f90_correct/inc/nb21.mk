#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb21  ########


nb21: run
	

build:  $(SRC)/nb21.f
	-$(RM) nb21.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb21.f -o nb21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb21.$(OBJX) check.$(OBJX) $(LIBS) -o nb21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb21
	nb21.$(EXESUFFIX)

verify: ;

nb21.run: run

