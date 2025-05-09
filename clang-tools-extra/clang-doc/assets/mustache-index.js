document.addEventListener("DOMContentLoaded", function() {
  const resizer = document.getElementById('resizer');
  const sidebar = document.querySelector('.sidebar');

  let isResizing = false;
  resizer.addEventListener('mousedown', (e) => { isResizing = true; });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing)
      return;
    const newWidth = e.clientX;
    if (newWidth > 100 && newWidth < window.innerWidth - 100) {
      sidebar.style.width = `${newWidth}px`;
    }
  });

  document.addEventListener('mouseup', () => { isResizing = false; });

  document.querySelectorAll('pre code').forEach((el) => {
    hljs.highlightElement(el);
    el.classList.remove("hljs");
  });

  document.querySelectorAll('.sidebar-item-container').forEach(item => {
    item.addEventListener('click', function() {
      const anchor = item.getElementsByTagName("a");
      window.location.hash = anchor[0].getAttribute('href');
    });
  });
})
