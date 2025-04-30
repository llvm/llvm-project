/* Prototype for the setgrent functions we use here.  */
typedef enum nss_status (*set_function) (void);

/* Prototype for the endgrent functions we use here.  */
typedef enum nss_status (*end_function) (void);

/* Prototype for the setgrent functions we use here.  */
typedef enum nss_status (*get_function) (struct group *, char *,
					 size_t, int *);


static enum nss_status
compat_call (nss_action_list nip, const char *user, gid_t group, long int *start,
	     long int *size, gid_t **groupsp, long int limit, int *errnop)
{
  struct group grpbuf;
  enum nss_status status;
  set_function setgrent_fct;
  get_function getgrent_fct;
  end_function endgrent_fct;
  gid_t *groups = *groupsp;

  getgrent_fct = __nss_lookup_function (nip, "getgrent_r");
  if (getgrent_fct == NULL)
    return NSS_STATUS_UNAVAIL;

  setgrent_fct = __nss_lookup_function (nip, "setgrent");
  if (setgrent_fct)
    {
      status = DL_CALL_FCT (setgrent_fct, ());
      if (status != NSS_STATUS_SUCCESS)
	return status;
    }

  endgrent_fct = __nss_lookup_function (nip, "endgrent");

  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);
  enum nss_status result = NSS_STATUS_SUCCESS;

  do
    {
      while ((status = DL_CALL_FCT (getgrent_fct,
				     (&grpbuf, tmpbuf.data, tmpbuf.length,
				      errnop)),
	      status == NSS_STATUS_TRYAGAIN)
	     && *errnop == ERANGE)
        {
	  if (!scratch_buffer_grow (&tmpbuf))
	    {
	      result = NSS_STATUS_TRYAGAIN;
	      goto done;
	    }
        }

      if (status != NSS_STATUS_SUCCESS)
        goto done;

      if (grpbuf.gr_gid != group)
        {
          char **m;

          for (m = grpbuf.gr_mem; *m != NULL; ++m)
            if (strcmp (*m, user) == 0)
              {
		/* Check whether the group is already on the list.  */
		long int cnt;
		for (cnt = 0; cnt < *start; ++cnt)
		  if (groups[cnt] == grpbuf.gr_gid)
		    break;

		if (cnt == *start)
		  {
		    /* Matches user and not yet on the list.  Insert
		       this group.  */
		    if (__glibc_unlikely (*start == *size))
		      {
			/* Need a bigger buffer.  */
			gid_t *newgroups;
			long int newsize;

			if (limit > 0 && *size == limit)
			  /* We reached the maximum.  */
			  goto done;

			if (limit <= 0)
			  newsize = 2 * *size;
			else
			  newsize = MIN (limit, 2 * *size);

			newgroups = realloc (groups,
					     newsize * sizeof (*groups));
			if (newgroups == NULL)
			  goto done;
			*groupsp = groups = newgroups;
			*size = newsize;
		      }

		    groups[*start] = grpbuf.gr_gid;
		    *start += 1;
		  }

                break;
              }
        }
    }
  while (status == NSS_STATUS_SUCCESS);

 done:
  scratch_buffer_free (&tmpbuf);

  if (endgrent_fct)
    DL_CALL_FCT (endgrent_fct, ());

  return result;
}
