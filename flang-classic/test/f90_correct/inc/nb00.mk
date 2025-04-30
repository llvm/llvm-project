#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb00  ########


nb00: run
	

build:  $(SRC)/nb00.f
	-$(RM) nb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb00.f -o nb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb00.$(OBJX) check.$(OBJX) $(LIBS) -o nb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb00
	nb00.$(EXESUFFIX)

verify: ;

nb00.run: run

