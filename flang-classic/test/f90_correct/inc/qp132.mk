#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp132  ########


qp132: run


build:  $(SRC)/qp132.f08
	-$(RM) qp132.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -Hx,57,0x10 -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp132.f08 -o qp132.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp132.$(OBJX) check.$(OBJX) $(LIBS) -o qp132.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp132
	qp132.$(EXESUFFIX)

verify: ;
