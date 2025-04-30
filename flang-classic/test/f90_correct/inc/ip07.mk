#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip07  ########


ip07: run
	

build:  $(SRC)/ip07.f90
	-$(RM) ip07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip07.f90 -o ip07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip07.$(OBJX) check.$(OBJX) $(LIBS) -o ip07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip07
	ip07.$(EXESUFFIX)

verify: ;

ip07.run: run

