#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test blogtoq  ########


qp72: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp72.f08 fcheck.$(OBJX)
	-$(RM) qp72.$(EXESUFFIX) qp72.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp72.f08 -o qp72.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp72.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp72.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp72
	qp72.$(EXESUFFIX)

verify: ;

qp72.run: run

