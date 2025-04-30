#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp24  ########


qp24: run


build:  $(SRC)/qp24.f08
	-$(RM) qp24.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp24.f08 -o qp24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp24.$(OBJX) check.$(OBJX) $(LIBS) -o qp24.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp24
	qp24.$(EXESUFFIX)

verify: ;
