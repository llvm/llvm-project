#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do12: do12.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do12.$(OBJX): $(SRC)/do12.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do12.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do12.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do12
run: ;
