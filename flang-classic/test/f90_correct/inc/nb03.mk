#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb03  ########


nb03: run
	

build:  $(SRC)/nb03.f
	-$(RM) nb03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb03.f -o nb03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb03.$(OBJX) check.$(OBJX) $(LIBS) -o nb03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb03
	nb03.$(EXESUFFIX)

verify: ;

nb03.run: run

